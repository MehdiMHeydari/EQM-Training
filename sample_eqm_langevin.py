"""
Sample from unconditional EQM model using Langevin dynamics.

Langevin dynamics adds noise during sampling to prevent mode collapse:
    x_{t+1} = x_t - α*∇E(x_t) + sqrt(2α)*ε_t

This maintains diversity by exploring the energy landscape instead of
converging to a single mode.
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


def sample_langevin_dynamics(model, num_samples, shape, device,
                              num_steps=500, step_size=0.002,
                              temperature=1.0, batch_size=16,
                              clip_range=(-1, 1)):
    """
    Sample using Langevin dynamics: x ← x - α*∇E(x) + sqrt(2α*T)*ε

    Args:
        model: Trained UNet model (energy function)
        num_samples: Number of samples to generate
        shape: Shape of each sample (C, H, W)
        device: torch device
        num_steps: Number of Langevin steps
        step_size: Step size α for Langevin dynamics
        temperature: Temperature T (higher = more diversity)
        batch_size: Batch size for sampling
        clip_range: Tuple (min, max) to clip samples

    Returns:
        samples: Generated solution fields as numpy array
    """
    model.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples (Langevin)"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Start from random noise
        x = torch.randn(current_batch_size, *shape).to(device)

        # Langevin dynamics
        for step in range(num_steps):
            x.requires_grad_(True)

            with torch.enable_grad():
                # Compute energy E(x) = sum(x * model(x))
                pred = model(x)
                E = torch.sum(x * pred, dim=(1, 2, 3))

                # Compute gradient ∇E(x)
                grad_E = torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

            # Langevin update: x ← x - α*∇E(x) + sqrt(2α*T)*ε
            with torch.no_grad():
                # Gradient descent step (negative because we want to minimize E)
                x = x - step_size * grad_E

                # Add Gaussian noise for exploration
                noise = torch.randn_like(x) * np.sqrt(2 * step_size * temperature)
                x = x + noise

                # Clip to valid range
                if clip_range is not None:
                    x = torch.clamp(x, clip_range[0], clip_range[1])

        all_samples.append(x.detach().cpu().numpy())

    return np.concatenate(all_samples, axis=0)[:num_samples]


def sample_annealed_langevin(model, num_samples, shape, device,
                              num_steps=500, initial_step_size=0.01,
                              final_step_size=0.0001, temperature=1.0,
                              batch_size=16, clip_range=(-1, 1)):
    """
    Sample using annealed Langevin dynamics with decreasing step size.

    Args:
        model: Trained UNet model (energy function)
        num_samples: Number of samples to generate
        shape: Shape of each sample (C, H, W)
        device: torch device
        num_steps: Number of Langevin steps
        initial_step_size: Initial step size (larger for exploration)
        final_step_size: Final step size (smaller for refinement)
        temperature: Temperature (higher = more diversity)
        batch_size: Batch size for sampling
        clip_range: Tuple (min, max) to clip samples

    Returns:
        samples: Generated solution fields as numpy array
    """
    model.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples (Annealed Langevin)"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Start from random noise
        x = torch.randn(current_batch_size, *shape).to(device)

        # Annealed Langevin dynamics
        for step in range(num_steps):
            # Anneal step size: large → small
            progress = step / num_steps
            step_size = initial_step_size * (1 - progress) + final_step_size * progress

            x.requires_grad_(True)

            with torch.enable_grad():
                pred = model(x)
                E = torch.sum(x * pred, dim=(1, 2, 3))
                grad_E = torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

            with torch.no_grad():
                x = x - step_size * grad_E

                # Add noise (also anneal noise magnitude)
                noise = torch.randn_like(x) * np.sqrt(2 * step_size * temperature)
                x = x + noise

                if clip_range is not None:
                    x = torch.clamp(x, clip_range[0], clip_range[1])

        all_samples.append(x.detach().cpu().numpy())

    return np.concatenate(all_samples, axis=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Sample from unconditional EQM using Langevin dynamics')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='samples_langevin.npy',
                       help='Output file for generated samples')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=500,
                       help='Number of Langevin steps')
    parser.add_argument('--step_size', type=float, default=0.002,
                       help='Step size for Langevin dynamics')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature (higher = more diversity, try 0.1-2.0)')
    parser.add_argument('--method', type=str, default='langevin',
                       choices=['langevin', 'annealed'],
                       help='Sampling method: langevin or annealed')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

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
        print(f"Loaded normalization stats: min={data_min:.4f}, max={data_max:.4f}")
        clip_range = (-1, 1)
    else:
        print("Warning: No normalization stats found in checkpoint.")
        clip_range = None

    sample_shape = tuple(config.unet.dim)
    print(f"Sample shape: {sample_shape}\n")

    # Generate samples
    print(f"Generating {args.num_samples} samples using {args.method} method...")
    print(f"  Steps: {args.num_steps}")
    print(f"  Step size: {args.step_size}")
    print(f"  Temperature: {args.temperature}")

    if args.method == 'langevin':
        samples = sample_langevin_dynamics(
            model=model,
            num_samples=args.num_samples,
            shape=sample_shape,
            device=device,
            num_steps=args.num_steps,
            step_size=args.step_size,
            temperature=args.temperature,
            batch_size=args.batch_size,
            clip_range=clip_range
        )
    else:  # annealed
        samples = sample_annealed_langevin(
            model=model,
            num_samples=args.num_samples,
            shape=sample_shape,
            device=device,
            num_steps=args.num_steps,
            initial_step_size=args.step_size * 5,  # Start with 5x larger
            final_step_size=args.step_size / 10,    # End with 10x smaller
            temperature=args.temperature,
            batch_size=args.batch_size,
            clip_range=clip_range
        )

    # Save samples
    print(f"\nSaving samples to {args.output}...")
    np.save(args.output, samples)

    print(f"\nDone! Generated {samples.shape[0]} samples with shape {samples.shape[1:]}")
    print(f"Sample statistics:")
    print(f"  Min:  {samples.min():.4f}")
    print(f"  Max:  {samples.max():.4f}")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Std:  {samples.std():.4f}")

    # Compute per-sample variance to check diversity
    sample_means = samples.reshape(samples.shape[0], -1).mean(axis=1)
    print(f"\nDiversity check:")
    print(f"  Std of sample means: {sample_means.std():.6f}")
    print(f"  (Higher = more diverse samples)")


if __name__ == "__main__":
    main()
