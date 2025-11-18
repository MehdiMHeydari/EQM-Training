"""
Energy-based regularization loss for neural operators using trained EQM model.

This module provides a physics-informed regularization loss that penalizes
out-of-distribution predictions from neural operators by using the energy
function learned by an Equilibrium Matching (EQM) model.

Usage:
    energy_loss = EnergyRegularizationLoss(checkpoint_path, config_path, device)

    # During neural operator training:
    prediction = neural_operator(input)
    mse_loss = F.mse_loss(prediction, target)
    energy_penalty = energy_loss(prediction)

    total_loss = 0.8 * mse_loss + 0.2 * energy_penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


class EnergyRegularizationLoss(nn.Module):
    """
    Energy-based regularization loss using a pre-trained EQM model.

    Encourages neural operator predictions to lie in the distribution
    of training data by penalizing high-energy states.

    Args:
        checkpoint_path: Path to trained EQM model checkpoint (.pth)
        config_path: Path to EQM training config (.yaml)
        device: torch device ('cuda' or 'cpu')
        target_energy: Target energy value (default: use mean energy of training data)
        temperature: Temperature parameter for scaling (default: 1.0)
        normalize_inputs: Whether to apply min-max normalization to inputs (default: True)
    """

    def __init__(self, checkpoint_path, config_path, device='cuda',
                 target_energy=None, temperature=1.0, normalize_inputs=True):
        super().__init__()

        self.device = torch.device(device)
        self.target_energy = target_energy
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

        print(f"Energy regularization loss initialized with target_energy={target_energy}, temperature={temperature}")

    def normalize(self, x):
        """Apply min-max normalization to [-1, 1] range."""
        if self.data_min is not None and self.data_max is not None:
            return 2 * (x - self.data_min) / (self.data_max - self.data_min + 1e-8) - 1
        return x

    def compute_energy(self, x):
        """
        Compute energy E(x) = sum(x * model(x)) for each sample.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            energies: Tensor of shape (B,) with energy for each sample
        """
        with torch.no_grad():
            pred = self.energy_model(x)

        # Energy: E(x) = sum(x * model(x)) over spatial dimensions
        # Shape: (B, C, H, W) -> (B,)
        energy = torch.sum(x * pred, dim=(1, 2, 3))

        return energy

    def forward(self, predictions, reduction='mean'):
        """
        Compute energy regularization loss.

        Args:
            predictions: Neural operator predictions (B, C, H, W)
            reduction: How to reduce batch ('mean', 'sum', or 'none')

        Returns:
            loss: Energy regularization loss
        """
        # Normalize inputs if needed
        if self.normalize_inputs:
            x_norm = self.normalize(predictions)
        else:
            x_norm = predictions

        # Compute energies
        energies = self.compute_energy(x_norm)

        # Compute loss based on energy
        if self.target_energy is not None:
            # Penalize deviation from target energy
            loss = (energies - self.target_energy) ** 2
        else:
            # Penalize high energy (want low energy = in-distribution)
            # Higher energy = more OOD = higher penalty
            loss = energies / self.temperature

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

        energies = self.compute_energy(x_norm)

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

    Args:
        checkpoint_path: Path to trained EQM model checkpoint
        config_path: Path to EQM config
        device: torch device
        mse_weight: Weight for MSE loss (default: 0.8)
        energy_weight: Weight for energy regularization (default: 0.2)
        target_energy: Target energy for regularization (optional)
        temperature: Temperature for energy scaling (default: 1.0)
        normalize_inputs: Whether to normalize inputs (default: True)
    """

    def __init__(self, checkpoint_path, config_path, device='cuda',
                 mse_weight=0.8, energy_weight=0.2,
                 target_energy=None, temperature=1.0,
                 normalize_inputs=True):
        super().__init__()

        self.mse_weight = mse_weight
        self.energy_weight = energy_weight

        # Initialize energy regularization
        self.energy_loss = EnergyRegularizationLoss(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            target_energy=target_energy,
            temperature=temperature,
            normalize_inputs=normalize_inputs
        )

        print(f"Combined loss initialized: {mse_weight}*MSE + {energy_weight}*Energy")

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

        # Energy regularization loss
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


# Example usage
if __name__ == "__main__":
    # Example: Create combined loss for neural operator training
    checkpoint_path = "experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth"
    config_path = "configs/darcy_flow_eqm.yaml"
    device = "cuda"

    # Initialize combined loss
    criterion = CombinedLoss(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        mse_weight=0.8,
        energy_weight=0.2,
        temperature=1.0
    )

    # Dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 128, 128, device=device)
    targets = torch.randn(batch_size, 1, 128, 128, device=device)

    # Compute loss
    loss, loss_dict = criterion(predictions, targets)

    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Get energy stats
    energy_stats = criterion.get_energy_stats(predictions)
    print("\nEnergy statistics:")
    for key, value in energy_stats.items():
        print(f"  {key}: {value:.2f}")
