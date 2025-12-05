"""
FIXED: Energy-based regularization loss with proper scaling

This module provides properly scaled energy regularization that:
1. Matches MSE magnitude (both in [0, 1] range typically)
2. Correctly penalizes OOD predictions
3. Rewards predictions with in-distribution energy

Three loss modes available:
- 'relative': Penalize deviation from training energy mean (recommended)
- 'threshold': Only penalize when energy exceeds threshold
- 'normalized': Normalize energy to [0, 1] range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


class EnergyRegularizationLoss(nn.Module):
    """
    Energy-based regularization loss using a pre-trained EQM model.

    FIXED VERSION with proper scaling and loss computation.
    """

    def __init__(self, checkpoint_path, config_path, device='cuda',
                 training_data_path=None, num_calibration_samples=100,
                 loss_mode='relative', temperature=1.0, normalize_inputs=True):
        """
        Args:
            checkpoint_path: Path to trained EQM model checkpoint (.pth)
            config_path: Path to EQM training config (.yaml)
            device: torch device ('cuda' or 'cpu')
            training_data_path: Path to training data for computing energy statistics
            num_calibration_samples: Number of samples to use for energy statistics
            loss_mode: How to compute loss from energy
                - 'relative': (energy - mean)^2 / (2 * std^2) - penalize deviation
                - 'threshold': ReLU(energy - threshold) / scale - only penalize OOD
                - 'normalized': (energy - min) / (max - min) - normalize to [0,1]
            temperature: Temperature parameter for scaling (default: 1.0)
            normalize_inputs: Whether to apply min-max normalization to inputs
        """
        super().__init__()

        self.device = torch.device(device)
        self.loss_mode = loss_mode
        self.temperature = temperature
        self.normalize_inputs = normalize_inputs

        # Load config
        config = OmegaConf.load(config_path)

        # Initialize EQM energy model
        print("Loading EQM energy model for regularization...")
        self.energy_model = UNetModel(
            dim=config.unet.dim,
            out_channels=config.unet.out_channels,
            num_channels=config.unet.num_channels,
            num_res_blocks=config.unet.res_blocks,
            channel_mult=config.unet.channel_mult,
            num_head_channels=config.unet.head_chans,
            attention_resolutions=config.unet.attn_res,
            dropout=config.unet.dropout,
            use_new_attention_order=config.unet.new_attn,
            use_scale_shift_norm=config.unet.film,
        )

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.energy_model.load_state_dict(checkpoint['model_state_dict'])
        self.energy_model.to(self.device)
        self.energy_model.eval()

        # Freeze energy model parameters
        for param in self.energy_model.parameters():
            param.requires_grad = False

        # Load normalization stats if available
        if 'normalization_stats' in checkpoint and normalize_inputs:
            norm_stats = checkpoint['normalization_stats']
            self.data_min = torch.tensor(norm_stats['data_min'], device=self.device)
            self.data_max = torch.tensor(norm_stats['data_max'], device=self.device)
            print(f"Loaded normalization stats: min={norm_stats['data_min']:.6f}, max={norm_stats['data_max']:.6f}")
        else:
            self.data_min = None
            self.data_max = None
            if normalize_inputs:
                print("WARNING: No normalization stats found. Assuming inputs are already normalized.")

        # Compute energy statistics from training data
        print(f"\nComputing energy statistics from training data...")
        self.energy_mean = None
        self.energy_std = None
        self.energy_min = None
        self.energy_max = None

        if training_data_path is not None:
            self._compute_energy_statistics(training_data_path, config, num_calibration_samples)
        else:
            print("WARNING: No training_data_path provided. Using default energy statistics.")
            print("  This may lead to poor scaling. Recommended to provide training data.")
            # Use rough estimates from previous experiments
            self.energy_mean = -51733.0
            self.energy_std = 3833.0
            self.energy_min = -60000.0
            self.energy_max = -40000.0

        # Print final statistics
        print(f"\nEnergy statistics:")
        print(f"  Mean: {self.energy_mean:.2f}")
        print(f"  Std:  {self.energy_std:.2f}")
        print(f"  Min:  {self.energy_min:.2f}")
        print(f"  Max:  {self.energy_max:.2f}")
        print(f"\nLoss mode: {loss_mode}")
        print(f"Temperature: {temperature}")

    def _compute_energy_statistics(self, data_path, config, num_samples):
        """Compute energy statistics from training data."""
        import h5py
        import numpy as np

        # Load training data
        with h5py.File(data_path, 'r') as f:
            data = np.array(f[config.dataloader.output_key]).astype(np.float32)[:num_samples]

        # Ensure channel dimension
        if len(data.shape) == 3:
            data = data[:, np.newaxis, :, :]

        # Normalize
        if self.data_min is not None and self.data_max is not None:
            data_normalized = 2 * (data - self.data_min.cpu().numpy()) / \
                             (self.data_max.cpu().numpy() - self.data_min.cpu().numpy() + 1e-8) - 1
        else:
            data_normalized = data

        # Compute energies
        data_tensor = torch.from_numpy(data_normalized).float().to(self.device)
        with torch.no_grad():
            energies = []
            batch_size = 16
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i+batch_size]
                batch_energies = self.compute_energy(batch, allow_grad=False)
                energies.append(batch_energies)
            energies = torch.cat(energies)

        # Store statistics
        self.energy_mean = energies.mean().item()
        self.energy_std = energies.std().item()
        self.energy_min = energies.min().item()
        self.energy_max = energies.max().item()

        print(f"Computed energy statistics from {num_samples} training samples:")
        print(f"  Mean: {self.energy_mean:.2f}")
        print(f"  Std:  {self.energy_std:.2f}")
        print(f"  Min:  {self.energy_min:.2f}")
        print(f"  Max:  {self.energy_max:.2f}")

    def normalize(self, x):
        """Apply min-max normalization to [-1, 1] range."""
        if self.data_min is not None and self.data_max is not None:
            return 2 * (x - self.data_min) / (self.data_max - self.data_min + 1e-8) - 1
        return x

    def compute_energy(self, x, allow_grad=True):
        """
        Compute energy E(x) = sum(x * model(x)) for each sample.

        Args:
            x: Input tensor (B, C, H, W)
            allow_grad: If True, allow gradients to flow through energy model.
                        This is needed for training. Set False for inference.

        Returns:
            energies: Tensor of shape (B,) with energy for each sample
        """
        if allow_grad:
            # Allow gradients to flow through for proper training signal
            # Note: energy_model parameters are frozen, so they won't be updated
            pred = self.energy_model(x)
        else:
            with torch.no_grad():
                pred = self.energy_model(x)

        # Energy: E(x) = sum(x * model(x)) over spatial dimensions
        # Shape: (B, C, H, W) -> (B,)
        energy = torch.sum(x * pred, dim=(1, 2, 3))

        return energy

    def forward(self, predictions, reduction='mean'):
        """
        Compute energy regularization loss with proper scaling.

        Args:
            predictions: Neural operator predictions (B, C, H, W)
            reduction: How to reduce batch ('mean', 'sum', or 'none')

        Returns:
            loss: Energy regularization loss (properly scaled to match MSE)
        """
        # Normalize inputs if needed
        if self.normalize_inputs:
            x_norm = self.normalize(predictions)
        else:
            x_norm = predictions

        # Compute energies
        energies = self.compute_energy(x_norm)

        # Compute loss based on mode
        if self.loss_mode == 'relative':
            # Penalize deviation from training energy mean
            # Normalized by variance to make it scale-invariant
            # This gives loss in roughly [0, 1] range for typical deviations
            deviation = energies - self.energy_mean
            loss = (deviation ** 2) / (2 * self.energy_std ** 2 * self.temperature)

        elif self.loss_mode == 'threshold':
            # Only penalize when energy exceeds threshold (mean + 2*std)
            # This allows some natural variation but catches outliers
            threshold = self.energy_mean + 2 * self.energy_std
            excess = energies - threshold
            loss = torch.relu(excess) / (self.energy_std * self.temperature)

        elif self.loss_mode == 'normalized':
            # Normalize energy to [0, 1] range based on training data range
            # 0 = at minimum (best), 1 = at maximum (worst)
            loss = (energies - self.energy_min) / (self.energy_max - self.energy_min + 1e-8)
            loss = loss / self.temperature

        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def get_energy_stats(self, predictions):
        """
        Get energy statistics for monitoring during training.

        Args:
            predictions: Neural operator predictions (B, C, H, W)

        Returns:
            dict with 'mean', 'std', 'min', 'max' energy values
        """
        if self.normalize_inputs:
            x_norm = self.normalize(predictions)
        else:
            x_norm = predictions

        energies = self.compute_energy(x_norm, allow_grad=False)  # No grad needed for monitoring

        return {
            'mean': energies.mean().item(),
            'std': energies.std().item(),
            'min': energies.min().item(),
            'max': energies.max().item()
        }


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of MSE and energy regularization.

    total_loss = mse_weight * MSE(pred, target) + energy_weight * Energy(pred)

    FIXED VERSION with proper energy scaling.
    """

    def __init__(self, checkpoint_path, config_path, device='cuda',
                 mse_weight=0.8, energy_weight=0.2,
                 training_data_path=None, num_calibration_samples=100,
                 loss_mode='relative', temperature=1.0,
                 normalize_inputs=True):
        """
        Args:
            checkpoint_path: Path to trained EQM model checkpoint
            config_path: Path to EQM config
            device: torch device
            mse_weight: Weight for MSE loss (default: 0.8)
            energy_weight: Weight for energy regularization (default: 0.2)
            training_data_path: Path to training data for energy statistics
            num_calibration_samples: Number of samples for calibration
            loss_mode: Energy loss mode ('relative', 'threshold', or 'normalized')
            temperature: Temperature for energy scaling (default: 1.0)
            normalize_inputs: Whether to normalize inputs (default: True)
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.energy_weight = energy_weight

        # Initialize energy regularization
        self.energy_loss = EnergyRegularizationLoss(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            training_data_path=training_data_path,
            num_calibration_samples=num_calibration_samples,
            loss_mode=loss_mode,
            temperature=temperature,
            normalize_inputs=normalize_inputs
        )

        print(f"\nCombined loss initialized:")
        print(f"  {mse_weight}*MSE + {energy_weight}*Energy({loss_mode})")
        print(f"  Temperature: {temperature}")

    def forward(self, predictions, targets):
        """
        Compute combined loss.

        Args:
            predictions: Neural operator predictions (B, C, H, W)
            targets: Ground truth targets (B, C, H, W)

        Returns:
            loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)

        # Energy regularization loss (now properly scaled!)
        energy_reg = self.energy_loss(predictions)

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.energy_weight * energy_reg

        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'energy_reg': energy_reg.item(),
        }

        return total_loss, loss_dict

    def get_energy_stats(self, predictions):
        """Get energy statistics for monitoring."""
        return self.energy_loss.get_energy_stats(predictions)


# Example usage and comparison
if __name__ == "__main__":
    import numpy as np

    checkpoint_path = "experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth"
    config_path = "configs/darcy_flow_eqm.yaml"
    data_path = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
    device = "cuda"

    print("="*70)
    print("TESTING ENERGY LOSS MODES")
    print("="*70)

    # Dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 128, 128, device=device) * 0.5  # Normalized to roughly [-1, 1]
    targets = torch.randn(batch_size, 1, 128, 128, device=device) * 0.5

    # Test each loss mode
    for mode in ['relative', 'threshold', 'normalized']:
        print(f"\n{'='*70}")
        print(f"Testing mode: {mode}")
        print(f"{'='*70}")

        criterion = CombinedLoss(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            mse_weight=0.8,
            energy_weight=0.2,
            training_data_path=data_path,
            num_calibration_samples=100,
            loss_mode=mode,
            temperature=1.0
        )

        # Compute loss
        loss, loss_dict = criterion(predictions, targets)

        print(f"\nLoss components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.6f}")

        # Get energy stats
        energy_stats = criterion.get_energy_stats(predictions)
        print(f"\nEnergy statistics:")
        for key, value in energy_stats.items():
            print(f"  {key}: {value:.2f}")

        print(f"\nEnergy loss value: {loss_dict['energy_reg']:.6f}")
        print(f"MSE loss value: {loss_dict['mse']:.6f}")
        print(f"Ratio (energy/mse): {loss_dict['energy_reg']/loss_dict['mse']:.2f}")
