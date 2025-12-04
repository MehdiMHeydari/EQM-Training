# Energy Loss Scaling Guide

## The Problem

Your energy values are **huge** compared to MSE:

| Metric | Typical Range | Example Value |
|--------|---------------|---------------|
| MSE Loss | [0, 0.1] | 0.0234 |
| Energy (raw) | [-60000, -40000] | -51,733 |
| Energy / Temperature | [-60000, -40000] | -51,733 ❌ |

**Original bug** (line 145 in `energy_regularization.py`):
```python
loss = energies / self.temperature
# = -51733 / 1.0
# = -51733  ❌ NEGATIVE!

# Combined loss becomes:
total_loss = 0.8 * 0.0234 + 0.2 * (-51733)
           = 0.01872 + (-10346.6)
           = -10346.58  ❌❌❌ HUGE NEGATIVE!
```

This tells the optimizer to **minimize** total loss → make energy more negative → **wrong direction!**

## Energy Statistics from Your Data

From OOD detection experiments:

```
Clean (in-distribution):     Energy = -51,733 ± 3,833
Noisy σ=0.1 (slightly OOD):  Energy = -50,171 (+3% increase)
Noisy σ=0.5 (very OOD):      Energy = -26,571 (+49% increase)
Noisy σ=1.0 (extremely OOD): Energy = +5,487  (+111% increase)
```

**Key insight**: Higher (less negative) energy = more OOD

## Three Fixed Loss Modes

### Mode 1: **'relative'** (RECOMMENDED)

Penalize deviation from training data energy mean.

**Formula:**
```python
loss = (energy - energy_mean)² / (2 * energy_std²)
```

**Example:**
```
Training stats: mean = -51,733, std = 3,833

Prediction with energy = -51,733 (perfect):
  loss = (0)² / (2 * 3833²) = 0.0  ✓ No penalty

Prediction with energy = -48,000 (slightly OOD):
  deviation = -48,000 - (-51,733) = +3,733
  loss = (3,733)² / (2 * 3833²) = 0.475  ✓ Small penalty

Prediction with energy = -30,000 (very OOD):
  deviation = -30,000 - (-51,733) = +21,733
  loss = (21,733)² / (2 * 3833²) = 16.08  ✓ Large penalty
```

**Scaling:**
- Loss ≈ 0 for in-distribution predictions
- Loss ≈ 0.5 for 1 std deviation away
- Loss ≈ 2.0 for 2 std deviations away
- **Comparable to MSE** which is typically 0.01 - 0.1

**When to use:**
- ✓ Default choice for most cases
- ✓ You want predictions to match training data energy distribution
- ✓ You have reliable energy statistics from training data

---

### Mode 2: **'threshold'**

Only penalize when energy exceeds a threshold (mean + 2σ).

**Formula:**
```python
threshold = energy_mean + 2 * energy_std
loss = ReLU(energy - threshold) / energy_std
```

**Example:**
```
Training stats: mean = -51,733, std = 3,833
Threshold = -51,733 + 2*3,833 = -44,067

Prediction with energy = -51,000:
  excess = -51,000 - (-44,067) = -6,933
  loss = ReLU(-6,933) = 0.0  ✓ No penalty (within acceptable range)

Prediction with energy = -40,000:
  excess = -40,000 - (-44,067) = +4,067
  loss = 4,067 / 3,833 = 1.06  ✓ Penalty for being OOD

Prediction with energy = -20,000:
  excess = -20,000 - (-44,067) = +24,067
  loss = 24,067 / 3,833 = 6.28  ✓ Large penalty
```

**Scaling:**
- Loss = 0 for anything within 2 std of mean
- Loss increases linearly beyond threshold
- Normalized by std to be comparable to MSE

**When to use:**
- ✓ You want to allow natural variation (within 2σ)
- ✓ Only penalize clear outliers
- ✓ More permissive than 'relative' mode

---

### Mode 3: **'normalized'**

Normalize energy to [0, 1] range based on training data range.

**Formula:**
```python
loss = (energy - energy_min) / (energy_max - energy_min)
```

**Example:**
```
Training stats: min = -60,000, max = -40,000

Prediction with energy = -60,000 (best possible):
  loss = (-60,000 - (-60,000)) / 20,000 = 0.0  ✓

Prediction with energy = -50,000 (middle):
  loss = (-50,000 - (-60,000)) / 20,000 = 0.5  ✓

Prediction with energy = -40,000 (worst in training):
  loss = (-40,000 - (-60,000)) / 20,000 = 1.0  ✓

Prediction with energy = -30,000 (OOD, worse than any training):
  loss = (-30,000 - (-60,000)) / 20,000 = 1.5  ✓ Penalty
```

**Scaling:**
- Loss in [0, 1] for in-distribution energies
- Loss > 1 for energies worse than worst training sample
- Directly comparable to MSE

**When to use:**
- ✓ You want simple, interpretable loss
- ✓ You want loss directly in [0, 1] range
- ✗ May be less robust to outliers in training data

---

## Comparison Table

| Mode | Loss Range | Behavior | Best For |
|------|------------|----------|----------|
| **relative** | [0, ∞) | Gaussian-like penalty around mean | Most cases, robust |
| **threshold** | [0, ∞) | Zero until threshold, then linear | Allowing natural variation |
| **normalized** | [0, ∞) | Linear scaling based on range | Simple, interpretable |

## Recommended Settings

### Starting Point (Balanced)
```python
CombinedLoss(
    mse_weight=0.8,
    energy_weight=0.2,
    loss_mode='relative',  # ← RECOMMENDED
    temperature=1.0,
    training_data_path='data/2D_DarcyFlow_beta1.0_Train.hdf5',
    num_calibration_samples=100
)
```

**Expected loss values:**
```
Epoch 1:  MSE=0.05,  Energy=0.8,   Total=0.20
Epoch 50: MSE=0.01,  Energy=0.3,   Total=0.07
```

### More MSE Focus
```python
CombinedLoss(
    mse_weight=0.9,      # ← Increased
    energy_weight=0.1,   # ← Decreased
    loss_mode='threshold',  # Only catch outliers
    temperature=1.0
)
```

### More Physics Focus
```python
CombinedLoss(
    mse_weight=0.7,      # ← Decreased
    energy_weight=0.3,   # ← Increased
    loss_mode='relative',
    temperature=0.5      # ← Sharper penalty
)
```

## Temperature Parameter

Controls the **sharpness** of the penalty:

```python
# temperature = 0.5 (sharper, more aggressive)
loss = deviation² / (2 * std² * 0.5)
     = 2 * deviation² / (2 * std²)
     = 2x the loss

# temperature = 1.0 (default, balanced)
loss = deviation² / (2 * std² * 1.0)
     = 1x the loss

# temperature = 2.0 (softer, more permissive)
loss = deviation² / (2 * std² * 2.0)
     = 0.5x the loss
```

**When to adjust:**
- Temperature < 1.0: Model is producing too many OOD predictions
- Temperature > 1.0: Energy penalty is too harsh, MSE not improving

## How to Calibrate

### Step 1: Compute Training Energy Statistics

The fixed version automatically computes this:

```python
criterion = CombinedLoss(
    ...,
    training_data_path='data/2D_DarcyFlow_beta1.0_Train.hdf5',
    num_calibration_samples=100  # Use 100 samples for speed
)

# Outputs:
# Energy statistics:
#   Mean: -51733.45
#   Std:  3833.21
#   Min:  -59821.33
#   Max:  -42156.78
```

### Step 2: Train and Monitor

Watch the loss components:

```
Epoch 1:
  Total: 0.2000 (mse=0.0500, energy_reg=0.8000)
  Energy stats: μ=-52000, σ=3500

Epoch 50:
  Total: 0.0700 (mse=0.0100, energy_reg=0.3000)
  Energy stats: μ=-51500, σ=3600
```

**Good signs:**
- ✓ MSE decreases steadily
- ✓ Energy_reg starts high (~0.5-2.0) and decreases
- ✓ Energy mean approaches training mean (-51,733)
- ✓ Energy std stays wide (2000-4000)

**Bad signs:**
- ✗ Energy_reg stays very high (>5.0) → reduce energy_weight
- ✗ MSE plateaus early → reduce energy_weight or increase temperature
- ✗ Energy std becomes narrow (<500) → mode collapse, reduce energy_weight

### Step 3: Tune Weights

Start with 0.8/0.2, then adjust based on results:

| Observation | Action |
|-------------|--------|
| MSE not improving | Decrease energy_weight to 0.1 |
| Predictions look unphysical | Increase energy_weight to 0.3 |
| Energy increasing during training | Increase energy_weight or decrease temperature |
| Total loss dominated by energy | Decrease energy_weight or increase temperature |

## Visualizing Loss Balance

After training, check if losses are balanced:

```python
import matplotlib.pyplot as plt

# Load training history
mse_losses = history['train_mse']
energy_losses = history['train_energy']

# Plot weighted contributions
plt.plot(0.8 * np.array(mse_losses), label='0.8 × MSE')
plt.plot(0.2 * np.array(energy_losses), label='0.2 × Energy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss Contribution')
plt.title('Loss Component Contributions')
plt.show()
```

**Ideal result**: Both curves should be roughly similar magnitude and both decreasing.

**If 0.8×MSE >> 0.2×Energy**: Energy regularization is too weak, increase energy_weight
**If 0.2×Energy >> 0.8×MSE**: Energy regularization is too strong, decrease energy_weight

## Summary

**Use the fixed version:**
- ✓ Properly scaled energy loss (comparable to MSE)
- ✓ Computes statistics from training data automatically
- ✓ Three loss modes to choose from
- ✓ Temperature control for fine-tuning

**Recommended:**
- Start with `loss_mode='relative'`, `mse_weight=0.8`, `energy_weight=0.2`
- Monitor loss components during training
- Adjust weights if losses are unbalanced
- Check energy statistics to ensure predictions stay in-distribution
