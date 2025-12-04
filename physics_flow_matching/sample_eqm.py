"""
Sample from a trained EQM model for Darcy Flow.

Usage:
    python physics_flow_matching/sample_eqm.py \
        --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
        --config configs/darcy_flow_eqm.yaml \
        --num_samples 16 \
        --output samples.npy
"""

import argparse
import numpy as np
import torch
from torchdiffeq import odeint
from omegaconf import OmegaConf
from tqdm import tqdm

from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


def sample_eqm(model, num_samples, device, t_span=None, batch_size=16):
    """
    Sample from EQM model by solving the ODE.

    For EQM, the velocity field is v(x) = ∇E(x), where E is the energy function.
    We integrate backwards from t=1 (data) to t=0 (noise).

    Args:
        model: Trained UNet model (energy function)
        num_samples: Number of samples to generate
        device: torch device
        t_span: Time span for ODE integration (default: [0, 1] with 100 steps)
        batch_size: Batch size for sampling

    Returns:
        samples: Generated samples as numpy array
    """
    if t_span is None:
        t_span = torch.linspace(0, 1, 100).to(device)

    model.eval()
    all_samples = []

    # Get sample shape from model config
    sample_shape = (1, 128, 128)  # Darcy flow: 1 channel, 128x128

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Start from noise at t=0
        x0 = torch.randn(current_batch_size, *sample_shape, device=device)

        # Define ODE function: dx/dt = ∇E(x)
        def ode_func(t, x):
            x.requires_grad_(True)
            with torch.enable_grad():
                # Compute energy E(x)
                pred = model(x)
                E = torch.sum(x * pred, dim=(1, 2, 3))

                # Compute velocity v = ∇E(x)
                v = torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

            return v

        # Solve ODE from t=0 to t=1
        with torch.no_grad():
            trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )

        # Take final state (t=1, which is data distribution)
        samples = trajectory[-1].cpu().numpy()
        all_samples.append(samples)

    return np.concatenate(all_samples, axis=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Sample from trained EQM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for sampling')
    parser.add_argument('--output', type=str, default='samples.npy',
                        help='Output file path (.npy)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--ode_steps', type=int, default=100,
                        help='Number of ODE integration steps')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    print("Initializing model...")
    channel_mult = tuple(map(int, config.unet.channel_mult.split(",")))
    attn_res = tuple(map(int, config.unet.attn_res.split(",")))

    model = UNetModel(
        dim=config.unet.dim,
        out_channels=config.unet.out_channels,
        num_channels=config.unet.num_channels,
        num_res_blocks=config.unet.res_blocks,
        channel_mult=channel_mult,
        num_head_channels=config.unet.head_chans,
        attention_resolutions=attn_res,
        dropout=config.unet.dropout,
        use_new_attention_order=config.unet.new_attn,
        use_scale_shift_norm=config.unet.film,
        class_cond=config.unet.class_cond if hasattr(config.unet, 'class_cond') else False,
        num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Generating {args.num_samples} samples...")

    # Generate samples
    t_span = torch.linspace(0, 1, args.ode_steps).to(device)
    samples = sample_eqm(
        model=model,
        num_samples=args.num_samples,
        device=device,
        t_span=t_span,
        batch_size=args.batch_size
    )

    # Save samples
    print(f"Saving samples to {args.output}...")
    np.save(args.output, samples)

    print(f"Done! Generated {samples.shape[0]} samples with shape {samples.shape[1:]}")
    print(f"Sample statistics: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}")


if __name__ == "__main__":
    main()
