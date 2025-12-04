"""
FNO Training with Energy Regularization for Darcy Flow

Integrates the trained EQM energy function as a regularization loss.
Uses 0.8*MSE + 0.2*Energy loss to keep predictions in-distribution.

Based on Neural-Solver-Library's FNO implementation.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add Neural-Solver-Library to path
sys.path.insert(0, './Neural-Solver-Library-main')

from models.FNO import Model as FNO
from energy_regularization import CombinedLoss


class Args:
    """Arguments for FNO model"""
    def __init__(self):
        # Model architecture
        self.model = 'FNO'
        self.n_hidden = 128
        self.n_heads = 8
        self.n_layers = 8
        self.mlp_ratio = 2
        self.act = 'gelu'
        self.dropout = 0.0

        # FNO specific
        self.modes = 12  # Number of Fourier modes

        # Data
        self.geotype = 'structured_2D'
        self.space_dim = 2
        self.fun_dim = 1  # Input: permeability field
        self.out_dim = 1  # Output: pressure/solution field
        self.shapelist = [128, 128]  # 128x128 grid

        # Position embedding
        self.unified_pos = 1
        self.ref = 8

        # Task
        self.task = 'steady'
        self.time_input = False
        self.normalize = False  # We handle normalization ourselves


def load_darcy_data(data_path, train_samples=800, val_samples=200):
    """
    Load Darcy flow data.

    Returns:
        (train_inputs, train_outputs, val_inputs, val_outputs)
        All normalized to [-1, 1] to match EQM training
    """
    print(f"\nLoading data from {data_path}...")

    with h5py.File(data_path, 'r') as f:
        print(f"Available keys: {list(f.keys())}")

        # Load permeability (input) and solution (output)
        if 'nu' in f.keys():
            inputs = np.array(f['nu']).astype(np.float32)
            outputs = np.array(f['tensor']).astype(np.float32)
        else:
            # Fallback: use solution as both input and output
            outputs = np.array(f['tensor']).astype(np.float32)
            inputs = outputs.copy()
            print("Warning: Using solution as input (no 'nu' key found)")

    # Ensure shape is (N, H, W)
    if len(inputs.shape) == 4:
        inputs = inputs.squeeze(1)
    if len(outputs.shape) == 4:
        outputs = outputs.squeeze(1)

    print(f"Raw data shapes: inputs={inputs.shape}, outputs={outputs.shape}")
    print(f"Input range: [{inputs.min():.6f}, {inputs.max():.6f}]")
    print(f"Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")

    # Normalize outputs to [-1, 1] (same as EQM training)
    # Store normalization stats for later use
    output_min = outputs.min()
    output_max = outputs.max()
    outputs_normalized = 2 * (outputs - output_min) / (output_max - output_min + 1e-8) - 1

    print(f"\nNormalization stats:")
    print(f"  Output min: {output_min:.6f}")
    print(f"  Output max: {output_max:.6f}")
    print(f"  Normalized range: [{outputs_normalized.min():.6f}, {outputs_normalized.max():.6f}]")

    # Normalize inputs too (standard normalization for permeability)
    input_mean = inputs.mean()
    input_std = inputs.std()
    inputs_normalized = (inputs - input_mean) / (input_std + 1e-8)

    # Reshape to (N, C, H, W) for FNO
    # FNO expects (B, N, C) where N = H*W, but we'll handle reshaping in forward pass
    inputs_normalized = inputs_normalized[:, np.newaxis, :, :]  # (N, 1, H, W)
    outputs_normalized = outputs_normalized[:, np.newaxis, :, :]  # (N, 1, H, W)

    # Split train/val
    train_inputs = torch.FloatTensor(inputs_normalized[:train_samples])
    train_outputs = torch.FloatTensor(outputs_normalized[:train_samples])
    val_inputs = torch.FloatTensor(inputs_normalized[train_samples:train_samples+val_samples])
    val_outputs = torch.FloatTensor(outputs_normalized[train_samples:train_samples+val_samples])

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_inputs)} samples")
    print(f"  Validation: {len(val_inputs)} samples")

    normalization_stats = {
        'output_min': float(output_min),
        'output_max': float(output_max),
        'input_mean': float(input_mean),
        'input_std': float(input_std)
    }

    return train_inputs, train_outputs, val_inputs, val_outputs, normalization_stats


def prepare_batch_for_fno(inputs, outputs, args):
    """
    Prepare batch in format expected by FNO.

    FNO expects:
        x: position grid (B, N, 2) where N = H*W
        fx: function values at positions (B, N, 1)

    Returns output in (B, N, 1) which we reshape back to (B, 1, H, W)
    """
    B, C, H, W = inputs.shape

    # Create position grid
    grid_x = torch.linspace(0, 1, H, device=inputs.device)
    grid_y = torch.linspace(0, 1, W, device=inputs.device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)
    pos = pos.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 2)

    # Flatten inputs
    fx = inputs.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)

    return pos, fx


def train_fno_with_energy(args):
    """Main training function"""

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    train_inputs, train_outputs, val_inputs, val_outputs, norm_stats = load_darcy_data(
        args.data_path,
        args.train_samples,
        args.val_samples
    )

    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize FNO model
    print("\nInitializing FNO model...")
    fno_args = Args()
    model = FNO(fno_args, s1=128, s2=128).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO model parameters: {num_params:,}")

    # Initialize combined loss (MSE + Energy regularization)
    print("\nInitializing combined loss (MSE + Energy regularization)...")
    print(f"Loss formula: {args.mse_weight}*MSE + {args.energy_weight}*Energy({args.energy_loss_mode})")
    criterion = CombinedLoss(
        checkpoint_path=args.eqm_checkpoint,
        config_path=args.eqm_config,
        device=device,
        mse_weight=args.mse_weight,
        energy_weight=args.energy_weight,
        training_data_path=args.data_path,
        num_calibration_samples=100,
        loss_mode=args.energy_loss_mode,
        temperature=args.energy_temperature,
        normalize_inputs=True
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_scheduler:
        scheduler = None
        print("Using constant learning rate (no scheduler)")
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        print("Using OneCycleLR scheduler")

    # Training history
    history = {
        'train_loss': [], 'train_mse': [], 'train_energy': [],
        'val_loss': [], 'val_mse': [], 'val_energy': [],
        'energy_mean': [], 'energy_std': []
    }

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_mse, train_energy = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Prepare batch for FNO
            pos, fx = prepare_batch_for_fno(inputs, targets, fno_args)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(pos, fx)  # (B, N, 1)

            # Reshape predictions back to (B, 1, H, W)
            B = predictions.shape[0]
            predictions = predictions.reshape(B, 128, 128, 1).permute(0, 3, 1, 2)

            # Combined loss (MSE + Energy regularization)
            loss, loss_dict = criterion(predictions, targets)

            # Backward pass
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Accumulate losses
            train_loss += loss_dict['total']
            train_mse += loss_dict['mse']
            train_energy += loss_dict['energy_reg']

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'mse': f"{loss_dict['mse']:.4f}",
                'energy': f"{loss_dict['energy_reg']:.2f}"
            })

        # Average training losses
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_energy /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_mse, val_energy = 0, 0, 0
        energy_means, energy_stds = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Prepare batch
                pos, fx = prepare_batch_for_fno(inputs, targets, fno_args)

                # Forward pass
                predictions = model(pos, fx)
                predictions = predictions.reshape(-1, 128, 128, 1).permute(0, 3, 1, 2)

                # Compute losses
                loss, loss_dict = criterion(predictions, targets)
                energy_stats = criterion.get_energy_stats(predictions)

                val_loss += loss_dict['total']
                val_mse += loss_dict['mse']
                val_energy += loss_dict['energy_reg']
                energy_means.append(energy_stats['mean'])
                energy_stds.append(energy_stats['std'])

        # Average validation losses
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_energy /= len(val_loader)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['train_energy'].append(train_energy)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_energy'].append(val_energy)
        history['energy_mean'].append(np.mean(energy_means))
        history['energy_std'].append(np.mean(energy_stds))

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train: loss={train_loss:.4f} (mse={train_mse:.4f}, energy={train_energy:.2f})")
        print(f"  Val:   loss={val_loss:.4f} (mse={val_mse:.4f}, energy={val_energy:.2f})")
        print(f"  Energy stats: μ={history['energy_mean'][-1]:.1f}, σ={history['energy_std'][-1]:.1f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'normalization_stats': norm_stats
            }, args.checkpoint_save_path.replace('.pth', '_best.pth'))
            print(f"  → Best model saved! (val_loss={val_loss:.4f})")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'normalization_stats': norm_stats
            }, args.checkpoint_save_path.replace('.pth', f'_epoch{epoch+1}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'history': history,
        'normalization_stats': norm_stats
    }, args.checkpoint_save_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.checkpoint_save_path}")

    # Plot training curves
    plot_training_curves(history, args.output_plot)

    return model, history


def plot_training_curves(history, output_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss (0.8*MSE + 0.2*Energy)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # MSE loss
    ax = axes[0, 1]
    ax.plot(history['train_mse'], label='Train MSE', linewidth=2)
    ax.plot(history['val_mse'], label='Val MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('MSE Loss (Data Fitting)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Energy regularization
    ax = axes[1, 0]
    ax.plot(history['train_energy'], label='Train Energy', linewidth=2, linestyle='--')
    ax.plot(history['val_energy'], label='Val Energy', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Energy Regularization', fontsize=12)
    ax.set_title('Energy Regularization Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Energy statistics
    ax = axes[1, 1]
    ax.plot(history['energy_mean'], label='Mean Energy', linewidth=2, color='#e74c3c')
    ax.fill_between(range(len(history['energy_mean'])),
                     np.array(history['energy_mean']) - np.array(history['energy_std']),
                     np.array(history['energy_mean']) + np.array(history['energy_std']),
                     alpha=0.3, color='#e74c3c', label='±1 Std')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Energy E(x)', fontsize=12)
    ax.set_title('Prediction Energy Over Training\n(Lower = More In-Distribution)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('FNO Training with Energy Regularization', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FNO with Energy Regularization')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Darcy flow HDF5 file')
    parser.add_argument('--train_samples', type=int, default=800,
                       help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=200,
                       help='Number of validation samples')

    # EQM checkpoint
    parser.add_argument('--eqm_checkpoint', type=str, required=True,
                       help='Path to trained EQM checkpoint')
    parser.add_argument('--eqm_config', type=str, required=True,
                       help='Path to EQM config file')

    # Loss weights
    parser.add_argument('--mse_weight', type=float, default=0.8,
                       help='Weight for MSE loss')
    parser.add_argument('--energy_weight', type=float, default=0.2,
                       help='Weight for energy regularization')
    parser.add_argument('--energy_loss_mode', type=str, default='relative',
                       choices=['relative', 'threshold', 'normalized'],
                       help='Energy loss mode: relative (penalize deviation from mean), '
                            'threshold (only penalize outliers), or normalized (scale to [0,1])')
    parser.add_argument('--energy_temperature', type=float, default=1.0,
                       help='Temperature for energy loss (lower = sharper penalty)')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--no_scheduler', action='store_true',
                       help='Disable learning rate scheduler (use constant LR)')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                       help='Gradient clipping')

    # Saving
    parser.add_argument('--checkpoint_save_path', type=str, default='fno_with_energy.pth',
                       help='Path to save model checkpoint')
    parser.add_argument('--output_plot', type=str, default='fno_training_curves.png',
                       help='Path to save training curves plot')
    parser.add_argument('--save_every', type=int, default=25,
                       help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("FNO TRAINING WITH ENERGY REGULARIZATION")
    print("="*70)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    # Train
    model, history = train_fno_with_energy(args)


if __name__ == "__main__":
    main()
