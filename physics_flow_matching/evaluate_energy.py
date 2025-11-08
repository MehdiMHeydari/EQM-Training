"""
Evaluate and compare energy values between generated samples and ground truth data.

Uses the same energy formulation as training: E(x) = sum(x * model(x))

Usage:
    python physics_flow_matching/evaluate_energy.py \
        --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
        --config configs/darcy_flow_eqm.yaml \
        --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
        --num_samples 100
"""

import argparse
import numpy as np
import torch
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt

from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel
from physics_flow_matching.sample_eqm import sample_eqm


def compute_energy(x, model, device, batch_size=32):
    """
    Compute energy E(x) = sum(x * model(x)) for a batch of samples.

    Args:
        x: Samples as numpy array (N, C, H, W)
        model: Trained UNet model
        device: torch device
        batch_size: Batch size for computation

    Returns:
        energies: Energy values for each sample (N,)
    """
    model.eval()
    energies = []

    num_samples = x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            x_batch = torch.from_numpy(x[start_idx:end_idx]).float().to(device)

            # Compute E(x) = sum(x * model(x))
            pred = model(x_batch)
            E = torch.sum(x_batch * pred, dim=(1, 2, 3))

            energies.append(E.cpu().numpy())

    return np.concatenate(energies)


def load_ground_truth(data_path, num_samples=None, normalize=True):
    """
    Load ground truth data from HDF5 file.

    Args:
        data_path: Path to HDF5 file
        num_samples: Number of samples to load (None = all)
        normalize: Whether to normalize data

    Returns:
        data: Ground truth samples (N, 1, H, W)
    """
    with h5py.File(data_path, 'r') as f:
        data = np.array(f['tensor']).astype(np.float32)

    # Ensure channel dimension
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]

    # Take subset if specified
    if num_samples is not None:
        data = data[:num_samples]

    # Normalize
    if normalize:
        mean = data.mean(axis=(0, 2, 3), keepdims=True)
        std = data.std(axis=(0, 2, 3), keepdims=True)
        data = (data - mean) / (std + 1e-8)

    return data


def plot_energy_comparison(gt_energies, gen_energies, output_path=None):
    """
    Plot histogram and statistics comparing ground truth and generated energies.

    Args:
        gt_energies: Ground truth energy values
        gen_energies: Generated energy values
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    ax = axes[0, 0]
    ax.hist(gt_energies, bins=50, alpha=0.7, label='Ground Truth', density=True)
    ax.hist(gen_energies, bins=50, alpha=0.7, label='Generated', density=True)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
    ax.set_title('Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[0, 1]
    ax.boxplot([gt_energies, gen_energies], labels=['Ground Truth', 'Generated'])
    ax.set_ylabel('Energy')
    ax.set_title('Energy Box Plot')
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 0]
    gt_sorted = np.sort(gt_energies)
    gen_sorted = np.sort(gen_energies)
    # Match lengths
    min_len = min(len(gt_sorted), len(gen_sorted))
    ax.scatter(gt_sorted[:min_len], gen_sorted[:min_len], alpha=0.5)
    lims = [min(gt_sorted.min(), gen_sorted.min()),
            max(gt_sorted.max(), gen_sorted.max())]
    ax.plot(lims, lims, 'r--', label='Perfect match')
    ax.set_xlabel('Ground Truth Energy (sorted)')
    ax.set_ylabel('Generated Energy (sorted)')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    Statistics:

    Ground Truth:
      Mean:   {gt_energies.mean():.4f}
      Std:    {gt_energies.std():.4f}
      Min:    {gt_energies.min():.4f}
      Max:    {gt_energies.max():.4f}
      Median: {np.median(gt_energies):.4f}

    Generated:
      Mean:   {gen_energies.mean():.4f}
      Std:    {gen_energies.std():.4f}
      Min:    {gen_energies.min():.4f}
      Max:    {gen_energies.max():.4f}
      Median: {np.median(gen_energies):.4f}

    Differences:
      Mean diff:   {abs(gt_energies.mean() - gen_energies.mean()):.4f}
      Std diff:    {abs(gt_energies.std() - gen_energies.std()):.4f}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate energy of generated vs ground truth samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ground truth HDF5 data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for energy computation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_plot', type=str, default=None,
                        help='Path to save comparison plot')
    parser.add_argument('--ode_steps', type=int, default=100,
                        help='Number of ODE steps for sampling')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Load ground truth data
    print(f"Loading ground truth data from {args.data_path}...")
    gt_data = load_ground_truth(
        args.data_path,
        num_samples=args.num_samples,
        normalize=config.dataloader.normalize if hasattr(config.dataloader, 'normalize') else True
    )
    print(f"Loaded {gt_data.shape[0]} ground truth samples")

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    t_span = torch.linspace(0, 1, args.ode_steps).to(device)
    gen_data = sample_eqm(
        model=model,
        num_samples=args.num_samples,
        device=device,
        t_span=t_span,
        batch_size=args.batch_size
    )

    # Compute energies for ground truth
    print("Computing energies for ground truth data...")
    gt_energies = compute_energy(gt_data, model, device, args.batch_size)

    # Compute energies for generated samples
    print("Computing energies for generated samples...")
    gen_energies = compute_energy(gen_data, model, device, args.batch_size)

    # Print statistics
    print("\n" + "="*60)
    print("ENERGY COMPARISON")
    print("="*60)
    print("\nGround Truth:")
    print(f"  Mean:   {gt_energies.mean():.6f}")
    print(f"  Std:    {gt_energies.std():.6f}")
    print(f"  Min:    {gt_energies.min():.6f}")
    print(f"  Max:    {gt_energies.max():.6f}")
    print(f"  Median: {np.median(gt_energies):.6f}")

    print("\nGenerated:")
    print(f"  Mean:   {gen_energies.mean():.6f}")
    print(f"  Std:    {gen_energies.std():.6f}")
    print(f"  Min:    {gen_energies.min():.6f}")
    print(f"  Max:    {gen_energies.max():.6f}")
    print(f"  Median: {np.median(gen_energies):.6f}")

    print("\nDifferences:")
    print(f"  Mean diff: {abs(gt_energies.mean() - gen_energies.mean()):.6f}")
    print(f"  Std diff:  {abs(gt_energies.std() - gen_energies.std()):.6f}")
    print("="*60)

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_energy_comparison(gt_energies, gen_energies, args.output_plot)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
