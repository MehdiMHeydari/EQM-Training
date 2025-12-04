"""
Test Out-of-Distribution Detection using Energy

This script demonstrates that the learned energy function can distinguish
between clean (in-distribution) and corrupted (out-of-distribution) samples.

Expected behavior:
- Clean samples should have LOW energy (they're from the training distribution)
- Noisy samples should have HIGHER energy (they're out-of-distribution)

This validates that the model learned a meaningful energy landscape.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel
from tqdm import tqdm
import h5py
import argparse


def compute_energy(samples, model, device, batch_size=16):
    """
    Compute energy E(x) = sum(x * model(x)) for each sample.

    Args:
        samples: numpy array (N, C, H, W)
        model: Trained EQM model
        device: torch device
        batch_size: Batch size for processing

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
            energy = torch.sum(batch * pred, dim=(1, 2, 3))

            energies.append(energy.cpu().numpy())

    return np.concatenate(energies)


def add_gaussian_noise(samples, noise_std):
    """
    Add Gaussian noise to samples.

    Args:
        samples: numpy array (N, C, H, W)
        noise_std: Standard deviation of noise to add

    Returns:
        noisy_samples: Corrupted samples
    """
    noise = np.random.randn(*samples.shape) * noise_std
    return samples + noise


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
    parser = argparse.ArgumentParser(description='Test OOD detection using energy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to ground truth HDF5 file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to test')
    parser.add_argument('--noise_levels', type=float, nargs='+',
                       default=[0.1, 0.3, 0.5, 1.0],
                       help='Noise standard deviations to test (e.g., 0.1 0.3 0.5 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='ood_detection_results.png',
                       help='Output plot filename')

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
        print(f"Loaded normalization stats: min={data_min:.6f}, max={data_max:.6f}\n")
    else:
        raise ValueError("No normalization stats in checkpoint!")

    # Load ground truth samples
    print(f"Loading {args.num_samples} ground truth samples...")
    clean_samples = load_and_normalize_ground_truth(
        args.data_path,
        config.dataloader.output_key,
        data_min,
        data_max,
        num_samples=args.num_samples
    )

    print(f"Clean samples shape: {clean_samples.shape}")
    print(f"Clean samples range: [{clean_samples.min():.6f}, {clean_samples.max():.6f}]\n")

    # Compute energy for clean samples
    print("Computing energy for CLEAN samples...")
    clean_energies = compute_energy(clean_samples, model, device)

    print(f"\nClean samples energy:")
    print(f"  Mean: {clean_energies.mean():.2f}")
    print(f"  Std:  {clean_energies.std():.2f}")
    print(f"  Min:  {clean_energies.min():.2f}")
    print(f"  Max:  {clean_energies.max():.2f}")

    # Test different noise levels
    all_energies = {'Clean (σ=0)': clean_energies}
    all_samples = {'Clean (σ=0)': clean_samples}

    print(f"\nTesting {len(args.noise_levels)} noise levels...")
    for noise_std in args.noise_levels:
        print(f"\nAdding Gaussian noise with σ={noise_std}...")

        # Add noise
        noisy_samples = add_gaussian_noise(clean_samples, noise_std)

        print(f"  Noisy samples range: [{noisy_samples.min():.6f}, {noisy_samples.max():.6f}]")

        # Compute energy
        noisy_energies = compute_energy(noisy_samples, model, device)

        print(f"  Energy: mean={noisy_energies.mean():.2f}, std={noisy_energies.std():.2f}")
        print(f"  Energy increase: {noisy_energies.mean() - clean_energies.mean():.2f} "
              f"({(noisy_energies.mean() - clean_energies.mean()) / abs(clean_energies.mean()) * 100:.1f}%)")

        all_energies[f'Noisy (σ={noise_std})'] = noisy_energies
        all_samples[f'Noisy (σ={noise_std})'] = noisy_samples

    # Create comprehensive visualization
    print("\nCreating visualization...")

    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.5, 1])

    # Top row: Sample images showing progression of noise
    ax_samples = plt.subplot(gs[0, :])
    ax_samples.axis('off')

    num_samples_to_show = len(all_samples)
    sample_width = 1.0 / num_samples_to_show

    for idx, (label, samples) in enumerate(all_samples.items()):
        ax_img = fig.add_axes([idx * sample_width + 0.05, 0.68, sample_width * 0.85, 0.22])
        im = ax_img.imshow(samples[0, 0], cmap='viridis')
        ax_img.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax_img.axis('off')
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    # Middle: Mean energy trend line
    ax_trend = plt.subplot(gs[1, 0])

    noise_levels = [0.0] + args.noise_levels
    mean_energies = [all_energies[list(all_energies.keys())[i]].mean()
                     for i in range(len(all_energies))]
    std_energies = [all_energies[list(all_energies.keys())[i]].std()
                    for i in range(len(all_energies))]

    ax_trend.plot(noise_levels, mean_energies, 'o-', linewidth=3, markersize=12,
                 color='#e74c3c', label='Mean Energy')
    ax_trend.fill_between(noise_levels,
                          np.array(mean_energies) - np.array(std_energies),
                          np.array(mean_energies) + np.array(std_energies),
                          alpha=0.3, color='#e74c3c', label='±1 Std')

    ax_trend.axhline(clean_energies.mean(), color='#2ecc71', linestyle='--',
                    linewidth=2, label='Clean Mean', alpha=0.7)
    ax_trend.set_xlabel('Noise Level (σ)', fontsize=13, fontweight='bold')
    ax_trend.set_ylabel('Mean Energy', fontsize=13, fontweight='bold')
    ax_trend.set_title('Energy vs Noise Level\n(Clear separation = Good OOD detection)',
                      fontsize=14, fontweight='bold', pad=15)
    ax_trend.legend(fontsize=11, loc='best')
    ax_trend.grid(True, alpha=0.4, linewidth=1.2)

    # Add arrows and annotations
    ax_trend.annotate('', xy=(noise_levels[-1], mean_energies[-1]),
                     xytext=(noise_levels[0], mean_energies[0]),
                     arrowprops=dict(arrowstyle='->', lw=2.5, color='black', alpha=0.5))
    ax_trend.text(0.5, 0.15, 'Increasing Noise →\nIncreasing Energy →',
                 transform=ax_trend.transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                 ha='center')

    # Middle right: Stacked/separated histograms for clarity
    ax_hist = plt.subplot(gs[1, 1])

    from scipy import stats

    # Use KDE for smoother visualization
    all_energy_values = np.concatenate([energies for energies in all_energies.values()])
    energy_range = np.linspace(all_energy_values.min(), all_energy_values.max(), 300)

    for idx, (label, energies) in enumerate(all_energies.items()):
        # Kernel density estimation for smooth curve
        kde = stats.gaussian_kde(energies)
        density = kde(energy_range)

        # Offset each distribution vertically for clarity
        offset = idx * 0.00003
        ax_hist.fill_between(energy_range, offset, density + offset,
                            alpha=0.7, color=colors[idx % len(colors)],
                            label=f'{label} (μ={energies.mean():.0f})')
        ax_hist.plot(energy_range, density + offset, color=colors[idx % len(colors)],
                    linewidth=2.5)

        # Mark mean
        ax_hist.axvline(energies.mean(), ymin=offset/0.00015, ymax=(offset + 0.00003)/0.00015,
                       color=colors[idx % len(colors)], linestyle='--', linewidth=2, alpha=0.8)

    ax_hist.set_xlabel('Energy E(x) = sum(x * model(x))', fontsize=13, fontweight='bold')
    ax_hist.set_ylabel('Density (offset for clarity)', fontsize=13, fontweight='bold')
    ax_hist.set_title('Energy Distributions (Separated for Clarity)\n(Left = In-Distribution, Right = Out-of-Distribution)',
                     fontsize=14, fontweight='bold', pad=15)
    ax_hist.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax_hist.grid(True, alpha=0.3, linewidth=1)

    # Bottom: Energy statistics table
    ax_table = plt.subplot(gs[2, :])
    ax_table.axis('off')

    table_data = []
    headers = ['Noise Level', 'Mean Energy', 'Std Energy', 'Energy Increase', '% Increase']

    for idx, (label, energies) in enumerate(all_energies.items()):
        noise_val = label.split('=')[1].rstrip(')')
        mean_e = energies.mean()
        std_e = energies.std()
        diff = mean_e - clean_energies.mean()
        pct = (diff / abs(clean_energies.mean())) * 100

        table_data.append([
            noise_val,
            f'{mean_e:.1f}',
            f'{std_e:.1f}',
            f'{diff:+.1f}',
            f'{pct:+.1f}%'
        ])

    table = ax_table.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center',
                           colWidths=[0.15, 0.2, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color code the rows
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i % len(colors)])
            cell.set_alpha(0.3)

    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    plt.suptitle('Out-of-Distribution Detection using Energy Function\n✓ Higher Noise → Higher Energy → Successful OOD Detection',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: OOD DETECTION RESULTS")
    print("="*70)

    print("\nEnergy statistics by noise level:")
    print(f"{'Noise Level':<20} {'Mean Energy':>15} {'Std Energy':>15} {'vs Clean':>15}")
    print("-"*70)

    for label, energies in all_energies.items():
        diff = energies.mean() - clean_energies.mean()
        diff_pct = (diff / abs(clean_energies.mean())) * 100
        print(f"{label:<20} {energies.mean():>15.2f} {energies.std():>15.2f} "
              f"{diff:>+12.2f} ({diff_pct:>+6.1f}%)")

    print("\n" + "="*70)

    # Check if OOD detection is working
    if all(all_energies[f'Noisy (σ={std})'].mean() > clean_energies.mean()
           for std in args.noise_levels):
        print("✓ SUCCESS: Energy increases with noise level!")
        print("  → Model correctly assigns higher energy to OOD samples")
        print("  → Energy function is suitable for anomaly/OOD detection")
    else:
        print("✗ WARNING: Some noisy samples have lower energy than clean samples")
        print("  → Energy function may not be reliable for OOD detection")

    print("="*70)


if __name__ == "__main__":
    main()
