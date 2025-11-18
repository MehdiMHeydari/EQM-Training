"""
Compare energy values between generated samples and ground truth data.

This script:
1. Loads a trained EQM model with normalization stats
2. Generates samples from noise
3. Loads ground truth data and applies same normalization
4. Computes energies for both: E(x) = sum(x * model(x))
5. Plots histogram comparing energy distributions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel
from tqdm import tqdm
import h5py
import argparse

def compute_energy(samples, model, device, batch_size=16, clamp_model_output=None):
    """
    Compute energy E(x) = sum(x * model(x)) for each sample.

    Args:
        samples: numpy array (N, C, H, W)
        model: Trained EQM model
        device: torch device
        batch_size: Batch size for processing
        clamp_model_output: Tuple (min, max) to clamp model(x) before energy calculation

    Returns:
        energies: numpy array (N,) of energy values
    """
    model.eval()
    energies = []

    num_samples = len(samples)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing energies"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch = torch.from_numpy(samples[start_idx:end_idx]).float().to(device)

            # Compute energy: E(x) = sum(x * model(x))
            pred = model(batch)

            # Clamp model output if specified
            if clamp_model_output is not None:
                pred = torch.clamp(pred, clamp_model_output[0], clamp_model_output[1])

            energy = torch.sum(batch * pred, dim=(1, 2, 3))

            energies.append(energy.cpu().numpy())

    return np.concatenate(energies)


def sample_from_noise(model, num_samples, shape, device,
                     num_steps=500, step_size=0.002,
                     clip_range=(-1, 1), batch_size=16):
    """
    Generate samples from noise using gradient descent.

    Args:
        model: Trained EQM model
        num_samples: Number of samples to generate
        shape: Shape (C, H, W)
        device: torch device
        num_steps: Number of gradient descent steps
        step_size: Step size for gradient descent
        clip_range: Tuple (min, max) to clip samples
        batch_size: Batch size for generation

    Returns:
        samples: numpy array (N, C, H, W)
    """
    model.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Start from random noise
        x = torch.randn(current_batch_size, *shape).to(device)

        # Gradient descent
        for step in range(num_steps):
            x.requires_grad_(True)

            with torch.enable_grad():
                pred = model(x)
                E = torch.sum(x * pred, dim=(1, 2, 3))
                grad = -torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

            with torch.no_grad():
                x = x + step_size * grad
                if clip_range is not None:
                    x = torch.clamp(x, clip_range[0], clip_range[1])

        all_samples.append(x.detach().cpu().numpy())

    return np.concatenate(all_samples, axis=0)[:num_samples]


def load_and_normalize_ground_truth(hdf5_path, output_key, data_min, data_max, num_samples=None):
    """
    Load ground truth data and apply min-max normalization.

    Args:
        hdf5_path: Path to HDF5 file
        output_key: Key for output data
        data_min: Minimum value for normalization
        data_max: Maximum value for normalization
        num_samples: Number of samples to load (None = all)

    Returns:
        normalized_data: numpy array (N, C, H, W) normalized to [-1, 1]
    """
    with h5py.File(hdf5_path, 'r') as f:
        data = np.array(f[output_key]).astype(np.float32)

    # Ensure channel dimension
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]

    # Take subset if requested
    if num_samples is not None:
        data = data[:num_samples]

    # Apply same min-max normalization as training
    normalized = 2 * (data - data_min) / (data_max - data_min + 1e-8) - 1

    return normalized


def main():
    parser = argparse.ArgumentParser(description='Compare energies of generated vs ground truth samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to ground truth HDF5 file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate and compare')
    parser.add_argument('--num_steps', type=int, default=500,
                       help='Number of gradient descent steps for sampling')
    parser.add_argument('--step_size', type=float, default=0.002,
                       help='Step size for gradient descent')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='energy_comparison.png',
                       help='Output plot filename')
    parser.add_argument('--clamp_model_output', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='Clamp model(x) to [MIN, MAX] before energy calculation (e.g., -5 5)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load config
    config = OmegaConf.load(args.config)

    # Initialize model
    print("Initializing model...")
    model = UNetModel(
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load normalization stats
    if 'normalization_stats' in checkpoint:
        norm_stats = checkpoint['normalization_stats']
        data_min = norm_stats['data_min']
        data_max = norm_stats['data_max']
        print(f"Loaded normalization stats: min={data_min:.6f}, max={data_max:.6f}")
        clip_range = (-1, 1)
    else:
        print("WARNING: No normalization stats in checkpoint!")
        data_min = None
        data_max = None
        clip_range = None

    sample_shape = tuple(config.unet.dim)
    print(f"Sample shape: {sample_shape}\n")

    # Generate samples
    print(f"Generating {args.num_samples} samples from noise...")
    print(f"  Steps: {args.num_steps}, Step size: {args.step_size}")
    generated_samples = sample_from_noise(
        model, args.num_samples, sample_shape, device,
        num_steps=args.num_steps, step_size=args.step_size,
        clip_range=clip_range
    )

    print(f"\nGenerated samples stats:")
    print(f"  Shape: {generated_samples.shape}")
    print(f"  Range: [{generated_samples.min():.6f}, {generated_samples.max():.6f}]")
    print(f"  Mean: {generated_samples.mean():.6f}")

    # Load and normalize ground truth
    print(f"\nLoading ground truth from {args.data_path}...")
    ground_truth = load_and_normalize_ground_truth(
        args.data_path,
        config.dataloader.output_key,
        data_min,
        data_max,
        num_samples=args.num_samples
    )

    print(f"Ground truth stats:")
    print(f"  Shape: {ground_truth.shape}")
    print(f"  Range: [{ground_truth.min():.6f}, {ground_truth.max():.6f}]")
    print(f"  Mean: {ground_truth.mean():.6f}")

    # Convert clamp_model_output to tuple if provided
    clamp_model_output = tuple(args.clamp_model_output) if args.clamp_model_output is not None else None
    if clamp_model_output is not None:
        print(f"\nModel output will be clamped to {clamp_model_output} for energy calculation")

    # Compute energies
    print("\nComputing energies for generated samples...")
    generated_energies = compute_energy(generated_samples, model, device, clamp_model_output=clamp_model_output)

    print("Computing energies for ground truth...")
    ground_truth_energies = compute_energy(ground_truth, model, device, clamp_model_output=clamp_model_output)

    # Print statistics
    print("\n" + "="*60)
    print("ENERGY STATISTICS")
    print("="*60)

    print("\nGenerated Samples:")
    print(f"  Mean energy:   {generated_energies.mean():.6f}")
    print(f"  Std energy:    {generated_energies.std():.6f}")
    print(f"  Min energy:    {generated_energies.min():.6f}")
    print(f"  Max energy:    {generated_energies.max():.6f}")

    print("\nGround Truth:")
    print(f"  Mean energy:   {ground_truth_energies.mean():.6f}")
    print(f"  Std energy:    {ground_truth_energies.std():.6f}")
    print(f"  Min energy:    {ground_truth_energies.min():.6f}")
    print(f"  Max energy:    {ground_truth_energies.max():.6f}")

    print("\nDifference:")
    print(f"  Mean diff:     {abs(generated_energies.mean() - ground_truth_energies.mean()):.6f}")
    print(f"  Std diff:      {abs(generated_energies.std() - ground_truth_energies.std()):.6f}")

    # Plot histogram with improved visualization
    print(f"\nCreating energy comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Full range comparison with separate bins for each distribution
    ax1 = axes[0, 0]

    # Use separate bins for each distribution to show detail
    gt_bins = np.linspace(ground_truth_energies.min(), ground_truth_energies.max(), 50)
    gen_bins = np.linspace(generated_energies.min(), generated_energies.max(), 50)

    ax1.hist(ground_truth_energies, bins=gt_bins, alpha=0.7, label='Ground Truth',
             color='#1f77b4', density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(generated_energies, bins=gen_bins, alpha=0.7, label='Generated',
             color='#ff7f0e', density=True, edgecolor='black', linewidth=0.5)

    ax1.axvline(ground_truth_energies.mean(), color='#1f77b4', linestyle='--',
                linewidth=2.5, label=f'GT Mean: {ground_truth_energies.mean():.1f}')
    ax1.axvline(generated_energies.mean(), color='#ff7f0e', linestyle='--',
                linewidth=2.5, label=f'Gen Mean: {generated_energies.mean():.1f}')

    ax1.set_xlabel('Energy E(x) = sum(x * model(x))', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Energy Distribution Comparison (Separate Bins)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed view of generated samples only
    ax2 = axes[0, 1]
    ax2.hist(generated_energies, bins=30, alpha=0.8, color='#ff7f0e',
             edgecolor='black', linewidth=0.5)
    ax2.axvline(generated_energies.mean(), color='darkred', linestyle='--',
                linewidth=2.5, label=f'Mean: {generated_energies.mean():.1f}')
    ax2.axvline(generated_energies.mean() + generated_energies.std(),
                color='red', linestyle=':', linewidth=2, alpha=0.7,
                label=f'±1 Std: {generated_energies.std():.1f}')
    ax2.axvline(generated_energies.mean() - generated_energies.std(),
                color='red', linestyle=':', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Energy', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Generated Samples (Zoomed)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot comparison
    ax3 = axes[1, 0]
    box_data = [ground_truth_energies, generated_energies]
    bp = ax3.boxplot(box_data, labels=['Ground Truth', 'Generated'],
                      patch_artist=True, widths=0.6,
                      medianprops=dict(color='red', linewidth=2.5),
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

    # Color the boxes differently
    bp['boxes'][0].set_facecolor('#1f77b4')
    bp['boxes'][1].set_facecolor('#ff7f0e')

    ax3.set_ylabel('Energy', fontsize=11)
    ax3.set_title('Distribution Statistics (Box Plot)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Overlapping histograms with common bins (original view)
    ax4 = axes[1, 1]
    all_energies = np.concatenate([generated_energies, ground_truth_energies])
    common_bins = np.linspace(all_energies.min(), all_energies.max(), 60)

    ax4.hist(ground_truth_energies, bins=common_bins, alpha=0.6,
             label='Ground Truth', color='#1f77b4', density=True,
             edgecolor='black', linewidth=0.3)
    ax4.hist(generated_energies, bins=common_bins, alpha=0.6,
             label='Generated', color='#ff7f0e', density=True,
             edgecolor='black', linewidth=0.3)

    ax4.set_xlabel('Energy', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Overlapping Histograms (Common Bins)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Energy Distribution Analysis (N={args.num_samples}, Min-Max Normalization [-1, 1])',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Final assessment
    mean_diff = abs(generated_energies.mean() - ground_truth_energies.mean())
    if mean_diff < 0.5:
        print("\n✓ Energy distributions are very close - model is well-trained!")
    elif mean_diff < 1.0:
        print("\n⚠ Energy distributions are reasonably close - model is okay")
    else:
        print("\n✗ Energy distributions differ significantly - model may need more training")


if __name__ == "__main__":
    main()
