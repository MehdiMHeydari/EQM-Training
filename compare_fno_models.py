"""
Compare FNO Models: MSE+Energy vs MSE-Only

This script compares two trained FNO models:
1. MSE+Energy: Trained with 0.8*MSE + 0.2*Energy regularization
2. MSE-Only: Trained with pure MSE loss

Metrics compared:
- MSE (prediction accuracy)
- Energy distribution (physical plausibility)
- Visual quality of predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Neural-Solver-Library-main'))

from models.FNO import Model as FNO
from energy_regularization import EnergyRegularizationLoss


def load_fno_model(checkpoint_path, device):
    """Load a trained FNO model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize FNO model
    model = FNO(
        img_size=(128, 128),
        patch_size=1,
        in_channels=1,
        out_channels=1,
        embed_dim=256,
        depth=12,
        modes=32,
        num_blocks=8
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def compute_metrics(model, inputs, targets, energy_loss, device):
    """Compute MSE and energy for model predictions."""
    with torch.no_grad():
        predictions = model(inputs)

        # MSE per sample
        mse_per_sample = ((predictions - targets) ** 2).mean(dim=(1, 2, 3))

        # Energy per sample
        energies = energy_loss.compute_energy(predictions)

    return {
        'predictions': predictions.cpu().numpy(),
        'mse_per_sample': mse_per_sample.cpu().numpy(),
        'mse_mean': mse_per_sample.mean().item(),
        'mse_std': mse_per_sample.std().item(),
        'energies': energies.cpu().numpy(),
        'energy_mean': energies.mean().item(),
        'energy_std': energies.std().item()
    }


def main():
    parser = argparse.ArgumentParser(description='Compare FNO models')
    parser.add_argument('--mse_energy_checkpoint', type=str, required=True,
                        help='Path to MSE+Energy model checkpoint')
    parser.add_argument('--mse_only_checkpoint', type=str, required=True,
                        help='Path to MSE-only model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 data file')
    parser.add_argument('--eqm_checkpoint', type=str, required=True,
                        help='Path to EQM checkpoint for energy computation')
    parser.add_argument('--eqm_config', type=str, required=True,
                        help='Path to EQM config')
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help='Number of test samples to evaluate')
    parser.add_argument('--output_plot', type=str, default='model_comparison.png',
                        help='Output path for comparison plot')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)

    print(f"\nLoading MSE+Energy model from: {args.mse_energy_checkpoint}")
    model_energy, ckpt_energy = load_fno_model(args.mse_energy_checkpoint, device)
    print(f"  Best epoch: {ckpt_energy['epoch']}, Val loss: {ckpt_energy['val_loss']:.4f}")

    print(f"\nLoading MSE-only model from: {args.mse_only_checkpoint}")
    model_mse, ckpt_mse = load_fno_model(args.mse_only_checkpoint, device)
    print(f"  Best epoch: {ckpt_mse['epoch']}, Val loss: {ckpt_mse['val_loss']:.4f}")

    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)

    with h5py.File(args.data_path, 'r') as f:
        # Use samples after train+val split (800+200=1000)
        start_idx = 1000
        end_idx = start_idx + args.num_test_samples

        # Check available samples
        total_samples = f['tensor'].shape[0]
        if end_idx > total_samples:
            end_idx = total_samples
            start_idx = max(0, end_idx - args.num_test_samples)

        test_data = np.array(f['tensor'][start_idx:end_idx]).astype(np.float32)

        # Also load inputs (a(x,y)) - index 0
        inputs_data = np.array(f['x-coordinate'][start_idx:end_idx]).astype(np.float32)

    print(f"Loaded {test_data.shape[0]} test samples")
    print(f"  Indices: {start_idx} to {end_idx}")

    # Normalize using MSE+Energy model's stats (both should have similar stats)
    norm_stats = ckpt_energy['normalization_stats']

    # Normalize outputs (targets)
    test_normalized = 2 * (test_data - norm_stats['output_min']) / \
                      (norm_stats['output_max'] - norm_stats['output_min']) - 1

    # Normalize inputs
    inputs_normalized = 2 * (inputs_data - norm_stats['input_min']) / \
                        (norm_stats['input_max'] - norm_stats['input_min']) - 1

    # Convert to tensors
    inputs_tensor = torch.from_numpy(inputs_normalized[:, np.newaxis, :, :]).float().to(device)
    targets_tensor = torch.from_numpy(test_normalized[:, np.newaxis, :, :]).float().to(device)

    print(f"Input shape: {inputs_tensor.shape}")
    print(f"Target shape: {targets_tensor.shape}")

    # Initialize energy loss for computing energies
    print("\n" + "="*60)
    print("Initializing Energy Computation")
    print("="*60)

    energy_loss = EnergyRegularizationLoss(
        checkpoint_path=args.eqm_checkpoint,
        config_path=args.eqm_config,
        device=device,
        training_data_path=args.data_path,
        num_calibration_samples=100,
        loss_mode='relative'
    )

    training_energy_mean = energy_loss.energy_mean
    training_energy_std = energy_loss.energy_std

    print(f"\nTraining data energy reference:")
    print(f"  Mean: {training_energy_mean:.2f}")
    print(f"  Std:  {training_energy_std:.2f}")

    # Compute ground truth energy
    gt_energies = energy_loss.compute_energy(targets_tensor).cpu().numpy()
    print(f"\nGround truth test energy:")
    print(f"  Mean: {gt_energies.mean():.2f}")
    print(f"  Std:  {gt_energies.std():.2f}")

    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating Models")
    print("="*60)

    print("\nEvaluating MSE+Energy model...")
    metrics_energy = compute_metrics(model_energy, inputs_tensor, targets_tensor, energy_loss, device)

    print("\nEvaluating MSE-only model...")
    metrics_mse = compute_metrics(model_mse, inputs_tensor, targets_tensor, energy_loss, device)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print("\n{:<25} {:>15} {:>15}".format("Metric", "MSE+Energy", "MSE-Only"))
    print("-" * 55)
    print("{:<25} {:>15.6f} {:>15.6f}".format(
        "MSE (mean)", metrics_energy['mse_mean'], metrics_mse['mse_mean']))
    print("{:<25} {:>15.6f} {:>15.6f}".format(
        "MSE (std)", metrics_energy['mse_std'], metrics_mse['mse_std']))
    print("{:<25} {:>15.2f} {:>15.2f}".format(
        "Energy (mean)", metrics_energy['energy_mean'], metrics_mse['energy_mean']))
    print("{:<25} {:>15.2f} {:>15.2f}".format(
        "Energy (std)", metrics_energy['energy_std'], metrics_mse['energy_std']))

    # Compute energy deviation from training mean
    energy_dev_energy = abs(metrics_energy['energy_mean'] - training_energy_mean)
    energy_dev_mse = abs(metrics_mse['energy_mean'] - training_energy_mean)

    print("\n{:<25} {:>15.2f} {:>15.2f}".format(
        "Energy deviation from μ", energy_dev_energy, energy_dev_mse))
    print("{:<25} {:>15.2f} {:>15.2f}".format(
        "Deviation (in σ units)", energy_dev_energy/training_energy_std, energy_dev_mse/training_energy_std))

    # Winner determination
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if metrics_mse['mse_mean'] < metrics_energy['mse_mean']:
        mse_winner = "MSE-Only"
        mse_improvement = (metrics_energy['mse_mean'] - metrics_mse['mse_mean']) / metrics_energy['mse_mean'] * 100
        print(f"\n✓ MSE Winner: {mse_winner} ({mse_improvement:.1f}% better MSE)")
    else:
        mse_winner = "MSE+Energy"
        mse_improvement = (metrics_mse['mse_mean'] - metrics_energy['mse_mean']) / metrics_mse['mse_mean'] * 100
        print(f"\n✓ MSE Winner: {mse_winner} ({mse_improvement:.1f}% better MSE)")

    if energy_dev_energy < energy_dev_mse:
        energy_winner = "MSE+Energy"
        print(f"✓ Energy Winner: {energy_winner} (closer to training distribution)")
    else:
        energy_winner = "MSE-Only"
        print(f"✓ Energy Winner: {energy_winner} (closer to training distribution)")

    # Interpretation
    print("\nInterpretation:")
    if mse_winner == "MSE-Only" and energy_winner == "MSE+Energy":
        print("  → Trade-off detected!")
        print("  → MSE-Only achieves lower error but predictions are OUT-OF-DISTRIBUTION")
        print("  → MSE+Energy has higher error but predictions are PHYSICALLY PLAUSIBLE")
        print("  → This suggests the energy regularization is working as intended")
    elif mse_winner == "MSE+Energy":
        print("  → MSE+Energy wins on both metrics!")
        print("  → Energy regularization helped learn better representations")
    else:
        print("  → MSE-Only wins on both metrics")
        print("  → Energy regularization may be too strong or not well calibrated")

    # Create visualization
    print("\n" + "="*60)
    print("Creating Visualization")
    print("="*60)

    fig = plt.figure(figsize=(20, 16))

    # 1. Energy Distribution Comparison
    ax1 = fig.add_subplot(2, 3, 1)

    # Plot histograms
    bins = np.linspace(
        min(gt_energies.min(), metrics_energy['energies'].min(), metrics_mse['energies'].min()),
        max(gt_energies.max(), metrics_energy['energies'].max(), metrics_mse['energies'].max()),
        50
    )

    ax1.hist(gt_energies, bins=bins, alpha=0.5, label=f'Ground Truth (μ={gt_energies.mean():.0f})',
             color='green', density=True)
    ax1.hist(metrics_energy['energies'], bins=bins, alpha=0.5,
             label=f'MSE+Energy (μ={metrics_energy["energy_mean"]:.0f})', color='blue', density=True)
    ax1.hist(metrics_mse['energies'], bins=bins, alpha=0.5,
             label=f'MSE-Only (μ={metrics_mse["energy_mean"]:.0f})', color='red', density=True)

    # Training reference
    ax1.axvline(training_energy_mean, color='black', linestyle='--', linewidth=2,
                label=f'Training μ={training_energy_mean:.0f}')
    ax1.axvspan(training_energy_mean - 2*training_energy_std,
                training_energy_mean + 2*training_energy_std,
                alpha=0.1, color='gray', label='Training ±2σ')

    ax1.set_xlabel('Energy', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Energy Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # 2. MSE Distribution
    ax2 = fig.add_subplot(2, 3, 2)

    mse_bins = np.linspace(0, max(metrics_energy['mse_per_sample'].max(),
                                   metrics_mse['mse_per_sample'].max()), 30)

    ax2.hist(metrics_energy['mse_per_sample'], bins=mse_bins, alpha=0.6,
             label=f'MSE+Energy (μ={metrics_energy["mse_mean"]:.4f})', color='blue')
    ax2.hist(metrics_mse['mse_per_sample'], bins=mse_bins, alpha=0.6,
             label=f'MSE-Only (μ={metrics_mse["mse_mean"]:.4f})', color='red')

    ax2.set_xlabel('MSE per Sample', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('MSE Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)

    # 3. MSE vs Energy Scatter
    ax3 = fig.add_subplot(2, 3, 3)

    ax3.scatter(metrics_energy['mse_per_sample'], metrics_energy['energies'],
                alpha=0.6, label='MSE+Energy', color='blue', s=30)
    ax3.scatter(metrics_mse['mse_per_sample'], metrics_mse['energies'],
                alpha=0.6, label='MSE-Only', color='red', s=30)

    ax3.axhline(training_energy_mean, color='black', linestyle='--', alpha=0.5,
                label=f'Training Energy Mean')
    ax3.axhspan(training_energy_mean - 2*training_energy_std,
                training_energy_mean + 2*training_energy_std,
                alpha=0.1, color='gray')

    ax3.set_xlabel('MSE', fontsize=12)
    ax3.set_ylabel('Energy', fontsize=12)
    ax3.set_title('MSE vs Energy Trade-off', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)

    # 4-6. Sample Predictions Comparison
    sample_indices = [0, 1, 2]  # Show 3 samples

    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(2, 3, plot_idx + 4)

        # Get predictions
        input_img = inputs_normalized[sample_idx]
        gt_img = test_normalized[sample_idx]
        pred_energy = metrics_energy['predictions'][sample_idx, 0]
        pred_mse = metrics_mse['predictions'][sample_idx, 0]

        # Create comparison grid: Input | GT | MSE+Energy | MSE-Only
        comparison = np.concatenate([
            input_img, gt_img, pred_energy, pred_mse
        ], axis=1)

        im = ax.imshow(comparison, cmap='viridis', aspect='equal')

        # Add vertical lines to separate images
        for x in [128, 256, 384]:
            ax.axvline(x, color='white', linewidth=2)

        # Labels
        ax.set_xticks([64, 192, 320, 448])
        ax.set_xticklabels(['Input a(x)', 'Ground Truth', 'MSE+Energy', 'MSE-Only'], fontsize=10)
        ax.set_yticks([])

        # MSE for this sample
        mse_e = metrics_energy['mse_per_sample'][sample_idx]
        mse_m = metrics_mse['mse_per_sample'][sample_idx]
        energy_e = metrics_energy['energies'][sample_idx]
        energy_m = metrics_mse['energies'][sample_idx]

        ax.set_title(f'Sample {sample_idx+1}  |  MSE: {mse_e:.4f} vs {mse_m:.4f}  |  Energy: {energy_e:.0f} vs {energy_m:.0f}',
                     fontsize=11, fontweight='bold')

    plt.suptitle('FNO Model Comparison: MSE+Energy Regularization vs MSE-Only',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {args.output_plot}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"""
    MSE+Energy Model:
      - MSE:    {metrics_energy['mse_mean']:.6f} ± {metrics_energy['mse_std']:.6f}
      - Energy: {metrics_energy['energy_mean']:.2f} ± {metrics_energy['energy_std']:.2f}
      - Energy deviation: {energy_dev_energy:.2f} ({energy_dev_energy/training_energy_std:.2f}σ from training mean)

    MSE-Only Model:
      - MSE:    {metrics_mse['mse_mean']:.6f} ± {metrics_mse['mse_std']:.6f}
      - Energy: {metrics_mse['energy_mean']:.2f} ± {metrics_mse['energy_std']:.2f}
      - Energy deviation: {energy_dev_mse:.2f} ({energy_dev_mse/training_energy_std:.2f}σ from training mean)

    Training Reference:
      - Energy: {training_energy_mean:.2f} ± {training_energy_std:.2f}

    Ground Truth Test:
      - Energy: {gt_energies.mean():.2f} ± {gt_energies.std():.2f}
    """)

    plt.show()


if __name__ == '__main__':
    main()
