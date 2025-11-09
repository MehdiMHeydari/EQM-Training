# Conditional EQM for Darcy Flow: Complete Explanation

## Problem Overview

We want to train a conditional Equilibrium Matching (EQM) model that learns the mapping:
```
a(x,y) → u(x,y)
```
where `a(x,y)` is the permeability field (input) and `u(x,y)` is the PDE solution (output).

## How EQM Works

### Training Phase

EQM uses an energy-based formulation:

1. **Data Interpolation**: Create noisy interpolations between input and output
   ```python
   xt = (1-t)*x0 + t*x1 + (1-t)*eps
   ```
   where:
   - `x0 = a(x,y)` (input permeability)
   - `x1 = u(x,y)` (output solution)
   - `t ~ U[0,1]` (interpolation parameter)
   - `eps ~ N(0,I)` (noise)

2. **Target Velocity**: Compute the target velocity field
   ```python
   ut = -λ*c(t) * dxt/dt
      = -λ*c(t) * (x1 - x0 - eps)
      = λ*c(t) * (eps - (x1 - x0))
   ```
   Note the **negative sign** - EQM uses negative velocities!

3. **Energy Training**: Train the model to match the velocity via energy gradients
   ```python
   E(x) = sum(x * model(x))      # Energy function
   ∇E(x) = predicted velocity    # Gradient of energy

   Loss = ||∇E(xt) - ut||^2      # Match predicted to target velocity
   ```

### Inference Phase

To generate `u(x,y)` from a new `a(x,y)`:

1. **Start from input**: Initialize `x = a(x,y)`

2. **Follow negative gradient**: Solve the ODE
   ```python
   dx/dτ = -∇E(x)    # NEGATIVE gradient!
   ```

   The negative sign is crucial because:
   - During training: `∇E(x) = -λ*c(t)*dxt/dt`
   - The gradient points **opposite** to the flow direction
   - To flow from x0 to x1, we must negate it

3. **Converge to output**: The solution converges to `u(x,y)`

## Key Fixes Applied

### Fix 1: Make EquilibriumMatching Conditional

**Problem**: Original code deleted `x0` and only used `x1`, making it unconditional.

**Solution**: Modified `compute_mu_t` to use both:
```python
# BEFORE (unconditional)
def compute_mu_t(self, x0, x1, t):
    del x0  # Deleted input!
    return t * x1

# AFTER (conditional)
def compute_mu_t(self, x0, x1, t):
    return (1-t)*x0 + t*x1  # Use both input and output
```

### Fix 2: Correct Sampling Direction

**Problem**: Sampling used `+∇E(x)`, giving inverted results.

**Solution**: Use negative gradient in ODE:
```python
# BEFORE (wrong direction)
v = torch.autograd.grad([E.sum()], [x], create_graph=False)[0]

# AFTER (correct direction)
v = -torch.autograd.grad([E.sum()], [x], create_graph=False)[0]
```

## Configuration

Set these in `configs/darcy_flow_eqm.yaml`:

```yaml
dataloader:
  input_key: "nu"        # Permeability a(x,y) from HDF5
  output_key: "tensor"   # Solution u(x,y) from HDF5
  normalize: True        # Standardize both fields
  use_eqm_format: False  # Use conditional format (x0=input, x1=output)
```

## Usage

### Training
```bash
python physics_flow_matching/train_scripts/train_unet_eqm.py configs/darcy_flow_eqm.yaml
```

### Sampling (Conditional)
```bash
python physics_flow_matching/sample_eqm_conditional.py \
    --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
    --config configs/darcy_flow_eqm.yaml \
    --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
    --num_samples 16 \
    --output samples.npy
```

This will:
1. Load input fields `a(x,y)` from the HDF5 file
2. Generate corresponding solutions `u(x,y)` using the trained model
3. Save results to `samples.npy`

## Mathematical Details

### Why the negative sign?

During training, we learn:
```
∇E(xt) = ut = λ*c(t) * (eps - (x1 - x0))
```

Taking the derivative of `xt`:
```
dxt/dt = d/dt[(1-t)*x0 + t*x1 + (1-t)*eps]
       = -x0 + x1 - eps
       = (x1 - x0) - eps
```

So:
```
∇E(xt) = λ*c(t) * (eps - (x1 - x0))
       = -λ*c(t) * ((x1 - x0) - eps)
       = -λ*c(t) * dxt/dt
```

The gradient is **negative** of the natural flow direction. Therefore, to flow from x0 to x1 during inference, we must follow `-∇E(x)`.

### Energy Landscape Interpretation

The model learns an energy landscape where:
- Low-energy regions correspond to valid (a, u) pairs
- The gradient `-∇E(x)` points towards these low-energy regions
- Starting from any permeability `a`, following `-∇E` leads to the corresponding solution `u`

This is why EQM can handle conditional generation: the energy landscape encodes the relationship between inputs and outputs.
