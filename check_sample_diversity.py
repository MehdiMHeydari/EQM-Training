"""
Quick diagnostic to check if generated samples are actually diverse or collapsed.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def check_diversity(samples_path, num_to_check=10):
    """Check if samples are diverse or all identical."""

    # Load samples
    samples = np.load(samples_path)
    print(f"Loaded samples: {samples.shape}")

    # Global statistics
    print(f"\nGlobal statistics:")
    print(f"  Min:  {samples.min():.6f}")
    print(f"  Max:  {samples.max():.6f}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Std:  {samples.std():.6f}")

    # Per-sample statistics
    sample_means = samples.reshape(samples.shape[0], -1).mean(axis=1)
    sample_stds = samples.reshape(samples.shape[0], -1).std(axis=1)

    print(f"\nPer-sample statistics:")
    print(f"  Std of sample means: {sample_means.std():.6f}")
    print(f"  Mean of sample stds: {sample_stds.mean():.6f}")
    print(f"  Std of sample stds:  {sample_stds.std():.6f}")

    # Pairwise differences
    num_to_check = min(num_to_check, len(samples))
    print(f"\nPairwise differences (first {num_to_check} samples):")

    diffs = []
    for i in range(num_to_check):
        for j in range(i+1, num_to_check):
            diff = np.abs(samples[i] - samples[j]).mean()
            diffs.append(diff)
            if i < 3 and j < 3:  # Print first few
                print(f"  |sample_{i} - sample_{j}| = {diff:.6f}")

    print(f"\n  Mean pairwise difference: {np.mean(diffs):.6f}")
    print(f"  Std pairwise difference:  {np.std(diffs):.6f}")

    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)

    if sample_means.std() < 1e-4:
        print("❌ SEVERE MODE COLLAPSE: All samples are nearly identical!")
        print("   → Use Langevin dynamics with temperature > 0")
    elif sample_means.std() < 0.01:
        print("⚠️  MODERATE MODE COLLAPSE: Samples are very similar")
        print("   → Increase temperature or reduce num_steps")
    else:
        print("✓ Samples show diversity")

    # Visualize first few samples
    num_vis = min(5, len(samples))
    fig, axes = plt.subplots(2, num_vis, figsize=(3*num_vis, 6))

    for i in range(num_vis):
        # Show sample
        if len(samples.shape) == 4 and samples.shape[1] == 1:
            img = samples[i, 0]
        else:
            img = samples[i]

        axes[0, i].imshow(img, cmap='viridis')
        axes[0, i].set_title(f'Sample {i}')
        axes[0, i].axis('off')

        # Show difference from first sample
        if i > 0:
            diff = np.abs(img - (samples[0, 0] if len(samples.shape) == 4 else samples[0]))
            axes[1, i].imshow(diff, cmap='hot')
            axes[1, i].set_title(f'|S{i} - S0| (mean={diff.mean():.4f})')
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, 'Reference', ha='center', va='center',
                           transform=axes[1, i].transAxes, fontsize=12)
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_diversity_check.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to sample_diversity_check.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check sample diversity')
    parser.add_argument('samples', type=str, help='Path to .npy file with samples')
    parser.add_argument('--num_check', type=int, default=10,
                       help='Number of samples to check pairwise')

    args = parser.parse_args()
    check_diversity(args.samples, args.num_check)
