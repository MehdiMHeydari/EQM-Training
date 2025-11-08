"""
Sample from a trained conditional EQM model for Darcy Flow.

For conditional flow matching, you provide input permeability fields a(x,y) and the model
generates corresponding solution fields u(x,y).

Usage:
    python physics_flow_matching/sample_eqm_conditional.py \
        --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
        --config configs/darcy_flow_eqm.yaml \
        --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
        --num_samples 16 \
        --output samples.npy
"""

import argparse
import numpy as np
import torch
import h5py
from torchdiffeq import odeint
from omegaconf import OmegaConf
from tqdm import tqdm

from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel


def sample_conditional_eqm(model, input_fields, device, t_span=None, batch_size=16):
    """
    Sample from conditional EQM model: given a(x,y), generate u(x,y).

    For conditional EQM, we flow from x0=a(x,y) at t=0 to x1=u(x,y) at t=1.
    The velocity field v(x) = ∇E(x) guides this transformation.

    Args:
        model: Trained UNet model (energy function)
        input_fields: Input permeability fields a(x,y) as numpy array (N, 1, H, W)
        device: torch device
        t_span: Time span for ODE integration (default: [0, 1] with 100 steps)
        batch_size: Batch size for sampling

    Returns:
        samples: Generated solution fields u(x,y) as numpy array
    """
    if t_span is None:
        t_span = torch.linspace(0, 1, 100).to(device)

    model.eval()
    all_samples = []

    num_samples = input_fields.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        # Start from input permeability fields at t=0
        x0 = torch.from_numpy(input_fields[start_idx:end_idx]).float().to(device)

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

        # Solve ODE from t=0 (input) to t=1 (output)
        with torch.no_grad():
            trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5
            )

        # Take final state (t=1, which is the solution u(x,y))
        samples = trajectory[-1].cpu().numpy()
        all_samples.append(samples)

    return np.concatenate(all_samples, axis=0)[:num_samples]


def load_input_fields(data_path, num_samples=None, normalize=True, indices=None):
    """
    Load input permeability fields a(x,y) from HDF5 file.

    Args:
        data_path: Path to HDF5 file
        num_samples: Number of samples to load (None = all)
        normalize: Whether to normalize data
        indices: Specific indices to load (overrides num_samples)

    Returns:
        input_fields: Permeability fields (N, 1, H, W)
    """
    with h5py.File(data_path, 'r') as f:
        if indices is not None:
            data = np.array(f['nu'][indices]).astype(np.float32)
        elif num_samples is not None:
            data = np.array(f['nu'][:num_samples]).astype(np.float32)
        else:
            data = np.array(f['nu']).astype(np.float32)

    # Ensure channel dimension
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]

    # Normalize
    if normalize:
        mean = data.mean(axis=(0, 2, 3), keepdims=True)
        std = data.std(axis=(0, 2, 3), keepdims=True)
        data = (data - mean) / (std + 1e-8)

    return data


def main():
    parser = argparse.ArgumentParser(description='Sample from trained conditional EQM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 file with input fields')
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
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated list of specific indices to use (e.g., "0,5,10")')

    args = parser.parse_args()

    # Parse indices if provided
    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(',')]
        print(f"Using specific indices: {indices}")

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

    # Load input permeability fields
    print(f"Loading input fields from {args.data_path}...")
    input_fields = load_input_fields(
        args.data_path,
        num_samples=args.num_samples if indices is None else None,
        normalize=config.dataloader.normalize if hasattr(config.dataloader, 'normalize') else True,
        indices=indices
    )
    print(f"Loaded {input_fields.shape[0]} input fields")

    # Generate samples
    print(f"Generating {input_fields.shape[0]} solution fields...")
    t_span = torch.linspace(0, 1, args.ode_steps).to(device)
    samples = sample_conditional_eqm(
        model=model,
        input_fields=input_fields,
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
