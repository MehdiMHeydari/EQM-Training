# Model Output Clamping for EQM

## Problem

Your EQM model is experiencing mode collapse where all samples converge to the same point. This happens because:

1. **Energy calculation**: `E(x) = sum(x * model(x))`
2. If `model(x)` outputs extreme values → extreme energies → extreme gradients
3. Extreme gradients → aggressive convergence → all samples end up at the same point

## Solution: Clamp model(x)

By clamping `model(x)` before computing energy, we:
- Bound the energy landscape
- Prevent extreme gradients
- Allow more diverse samples to be stable

## How to Use

### 1. Sample with Clamping

```bash
python physics_flow_matching/sample_eqm_unconditional.py \
    --checkpoint path/to/checkpoint.pt \
    --config configs/darcy_flow_eqm.yaml \
    --output samples_clamped.npy \
    --num_samples 100 \
    --num_steps 500 \
    --step_size 0.002 \
    --clamp_model_output -5 5
```

### 2. Compare Energies with Clamping

**Important**: Use the SAME clamping values when comparing energies!

```bash
python compare_energies.py \
    --checkpoint path/to/checkpoint.pt \
    --config configs/darcy_flow_eqm.yaml \
    --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
    --num_samples 100 \
    --clamp_model_output -5 5 \
    --output energy_comparison_clamped.png
```

## Recommended Clamping Values

Start conservative and adjust based on results:

1. **First try: `--clamp_model_output -5 5`**
   - Moderate clamping
   - Should prevent extreme gradients
   - Good starting point

2. **If still collapsed: `--clamp_model_output -2 2`**
   - Stronger clamping
   - Forces gentler gradients
   - May help with severe collapse

3. **If too diverse/noisy: `--clamp_model_output -10 10`**
   - Lighter clamping
   - Allows stronger gradients
   - Better for quality

## How to Find Good Values

1. **Check unclamped model outputs**:
   ```python
   import torch
   import numpy as np

   # Load model and some samples
   pred = model(samples)
   print(f"Model output range: [{pred.min():.2f}, {pred.max():.2f}]")
   print(f"Model output mean: {pred.mean():.2f}")
   print(f"Model output std: {pred.std():.2f}")
   ```

2. **Set clamp range to ~2-3 standard deviations**:
   - If std = 3, try `--clamp_model_output -9 9` (3 × 3)
   - If std = 2, try `--clamp_model_output -6 6` (3 × 2)

## Expected Results

### Before Clamping:
```
Generated energies:
  Mean: -60775.797
  Std:  26.554          ← Very narrow!

Ground truth energies:
  Mean: -57054.031
  Std:  3833.068
```

### After Clamping (Hopefully):
```
Generated energies:
  Mean: -57000±1000     ← Closer to ground truth
  Std:  2000-4000       ← Much wider distribution!

Ground truth energies:
  Mean: -57054.031
  Std:  3833.068
```

## Troubleshooting

### Still seeing mode collapse?

1. Try **stronger clamping** (smaller range like `-2 2`)
2. Try **Langevin dynamics** instead (adds noise during sampling):
   ```bash
   python sample_eqm_langevin.py \
       --checkpoint path/to/checkpoint.pt \
       --config configs/darcy_flow_eqm.yaml \
       --temperature 1.0
   ```

### Samples look too noisy/bad quality?

1. Try **weaker clamping** (larger range like `-10 10`)
2. Reduce `--num_steps` or `--step_size`

### Generated energy still lower than ground truth?

This suggests the model learned a spurious attractor. You may need to:
1. Try even stronger clamping
2. Retrain with clamping applied during training (see next section)

## Advanced: Clamping During Training

If sampling-time clamping doesn't fully solve it, you can apply clamping during training too.

This requires modifying the training loop to clamp `model(x)` before computing the loss.

Let me know if you want to try this!

## Quick Diagnostic

To check if your current samples have mode collapse:

```bash
python check_sample_diversity.py samples_unconditional.npy
```

Look for:
- "Std of sample means" < 0.01 → severe collapse
- "Mean pairwise difference" < 0.001 → samples are identical
