"""
Sample from a trained unconditional EQM model for Darcy Flow.

For unconditional generation, the model generates solution fields u(x,y) from random noise,
without requiring input permeability fields a(x,y).

Usage:
    python physics_flow_matching/sample_eqm_unconditional.py \
        --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
        --config configs/darcy_flow_eqm.yaml \
        --num_samples 16 \
        --output samples_unconditional.npy
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


def sample_unconditional_gradient_descent(model, num_samples, shape, device,
                                          num_steps=100, step_size=0.01, batch_size=16,
                                          clip_range=(-1, 1)):
    """
    Sample from unconditional EQM model: noise → u(x,y).

    Args:
        model: Trained UNet model (energy function)
        num_samples: Number of samples to generate
        shape: Shape of each sample (C, H, W)
        device: torch device
        num_steps: Number of gradient descent steps
        step_size: Step size for gradient descent
        batch_size: Batch size for sampling
        clip_range: Tuple (min, max) to clip samples during generation

    Returns:
        samples: Generated solution fields u(x,y) as numpy array
    """
    model.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Start from random Gaussian noise
        x = torch.randn(current_batch_size, *shape).to(device)

        # Gradient descent loop
        for step in range(num_steps):
            x.requires_grad_(True)

            with torch.enable_grad():
                # Compute energy E(x) = sum(x * model(x))
                pred = model(x)
                E = torch.sum(x * pred, dim=(1, 2, 3))

                # Compute gradient v = -∇E(x)
                # EQM uses negative gradient for correct flow direction
                grad = -torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

            # Update x (gradient descent step)
            with torch.no_grad():
                x = x + step_size * grad
                # Clip to valid range to prevent drift out of distribution
                if clip_range is not None:
                    x = torch.clamp(x, clip_range[0], clip_range[1])

        # Save final samples
        all_samples.append(x.detach().cpu().numpy())

    return np.concatenate(all_samples, axis=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Sample from trained unconditional EQM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for sampling')
    parser.add_argument('--output', type=str, default='samples_unconditional.npy',
                        help='Output file path (.npy)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of gradient descent steps')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Gradient descent step size')

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

    # Load normalization stats from checkpoint
    if 'normalization_stats' in checkpoint:
        norm_stats = checkpoint['normalization_stats']
        data_min = norm_stats['data_min']
        data_max = norm_stats['data_max']
        print(f"Loaded normalization stats: min={data_min:.4f}, max={data_max:.4f}")
        print(f"Samples will be clipped to [-1, 1] during generation")
        clip_range = (-1, 1)
    else:
        print("Warning: No normalization stats found in checkpoint. Samples may drift out of distribution.")
        clip_range = None

    # Sample shape from config
    sample_shape = tuple(config.unet.dim)  # (C, H, W)
    print(f"Sample shape: {sample_shape}")

    # Generate samples
    print(f"Generating {args.num_samples} unconditional samples from noise...")
    samples = sample_unconditional_gradient_descent(
        model=model,
        num_samples=args.num_samples,
        shape=sample_shape,
        device=device,
        num_steps=args.num_steps,
        step_size=args.step_size,
        batch_size=args.batch_size,
        clip_range=clip_range
    )

    # Save samples
    print(f"Saving samples to {args.output}...")
    np.save(args.output, samples)

    print(f"Done! Generated {samples.shape[0]} samples with shape {samples.shape[1:]}")
    print(f"Sample statistics: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}")


if __name__ == "__main__":
    main()
