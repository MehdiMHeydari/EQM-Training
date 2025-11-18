# Energy Loss Scaling - Critical Fix Summary

## The Bug You Asked About

You asked: **"how do you determine the loss value from energy like how do you make sure its in line with MSE"**

**Answer**: The original implementation was BROKEN! ❌

## Original Code (WRONG)

```python
# Line 145 in energy_regularization.py
loss = energies / self.temperature
```

**Example values:**
```
Energy for in-distribution sample: -51,733
MSE for same sample: 0.0234

Original energy loss: -51,733 / 1.0 = -51,733  ❌ HUGE NEGATIVE!

Combined loss:
total_loss = 0.8 × 0.0234 + 0.2 × (-51,733)
           = 0.01872 + (-10,346.6)
           = -10,346.58  ❌❌❌ DISASTER!
```

**What went wrong:**
1. Energy values are large negative numbers (-60,000 to -40,000)
2. Dividing by temperature doesn't fix the scale
3. Total loss becomes hugely negative
4. Optimizer tries to minimize → makes energy MORE negative
5. **Wrong optimization direction!**

## Fixed Code (CORRECT)

I created **3 different loss modes** you can choose from:

### Mode 1: 'relative' (RECOMMENDED)

Penalizes deviation from training data mean:

```python
loss = (energy - energy_mean)² / (2 * energy_std²)
```

**Example:**
```
Training: mean = -51,733, std = 3,833

Perfect prediction (energy = -51,733):
  loss = 0² / (2 × 3,833²) = 0.0  ✓

Slightly OOD (energy = -48,000):
  loss = 3,733² / (2 × 3,833²) = 0.475  ✓ Comparable to MSE!

Very OOD (energy = -30,000):
  loss = 21,733² / (2 × 3,833²) = 16.08  ✓ Large penalty
```

### Mode 2: 'threshold'

Only penalizes outliers beyond 2σ:

```python
threshold = energy_mean + 2 * energy_std  # = -44,067
loss = ReLU(energy - threshold) / energy_std
```

**Example:**
```
Prediction with energy = -51,000 (within 2σ):
  loss = 0.0  ✓ No penalty

Prediction with energy = -40,000 (beyond 2σ):
  loss = (-40,000 - (-44,067)) / 3,833 = 1.06  ✓
```

### Mode 3: 'normalized'

Scales energy to [0, 1] based on training range:

```python
loss = (energy - energy_min) / (energy_max - energy_min)
```

**Example:**
```
Training: min = -60,000, max = -40,000

Best possible (energy = -60,000):
  loss = 0.0  ✓

Middle (energy = -50,000):
  loss = 0.5  ✓

Worst in training (energy = -40,000):
  loss = 1.0  ✓
```

## How Losses are Now Properly Scaled

| Loss Component | Typical Range | Example Value |
|----------------|---------------|---------------|
| MSE | [0, 0.1] | 0.0234 |
| Energy (relative) | [0, 5] | 0.475 |
| Energy (threshold) | [0, 3] | 1.06 |
| Energy (normalized) | [0, 1] | 0.5 |

**Now they're comparable!** ✓

## Combined Loss Example (Fixed)

```python
# With 'relative' mode
prediction_mse = 0.0234
prediction_energy_loss = 0.475

total_loss = 0.8 × 0.0234 + 0.2 × 0.475
           = 0.01872 + 0.095
           = 0.1137  ✓ CORRECT!
```

**Both losses contribute meaningfully!**

## How to Use the Fixed Version

### Quick Start (Default Settings)

```bash
./run_fno_training.sh
```

This uses:
- `loss_mode='relative'` (recommended)
- `mse_weight=0.8`
- `energy_weight=0.2`
- `temperature=1.0`

### Custom Settings

```bash
python train_fno_with_energy.py \
    --mse_weight 0.7 \
    --energy_weight 0.3 \
    --energy_loss_mode threshold \  # ← Choose mode
    --energy_temperature 0.5 \      # ← Lower = sharper penalty
    --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
    --eqm_checkpoint experiments/.../checkpoint_100.pth \
    --eqm_config configs/darcy_flow_eqm.yaml
```

## What's Automatically Computed

The fixed version automatically:

1. **Loads 100 training samples**
2. **Computes energy for each sample**
3. **Calculates statistics:**
   - Mean: -51,733
   - Std: 3,833
   - Min: -60,000
   - Max: -40,000

4. **Uses these for scaling**

Example output:
```
Computing energy statistics from training data...
Computed energy statistics from 100 training samples:
  Mean: -51733.45
  Std:  3833.21
  Min:  -59821.33
  Max:  -42156.78

Combined loss initialized:
  0.8*MSE + 0.2*Energy(relative)
  Temperature: 1.0
```

## Expected Training Output

```
Epoch 1/100:
  Train: loss=0.2000 (mse=0.0500, energy_reg=0.8000)
  Val:   loss=0.1900 (mse=0.0480, energy_reg=0.7500)
  Energy stats: μ=-52000, σ=3500

Epoch 50/100:
  Train: loss=0.0700 (mse=0.0100, energy_reg=0.3000)
  Val:   loss=0.0710 (mse=0.0105, energy_reg=0.3025)
  Energy stats: μ=-51500, σ=3600
  → Best model saved!
```

**Good signs:**
- ✓ Total loss is positive and decreasing
- ✓ MSE decreases from 0.05 → 0.01
- ✓ Energy_reg decreases from 0.8 → 0.3
- ✓ Both losses are comparable magnitude
- ✓ Energy mean approaches training mean

## Tuning Guide

### If MSE isn't improving:
```python
--mse_weight 0.9  # Increase
--energy_weight 0.1  # Decrease
```

### If predictions look unphysical:
```python
--mse_weight 0.7  # Decrease
--energy_weight 0.3  # Increase
```

### If energy is increasing during training:
```python
--energy_temperature 0.5  # Sharper penalty
# or
--energy_weight 0.3  # Stronger regularization
```

## Files Changed

1. **`energy_regularization.py`** - Fixed with proper scaling
2. **`train_fno_with_energy.py`** - Updated to use calibration
3. **`run_fno_training.sh`** - Added `--energy_loss_mode` flag
4. **`ENERGY_LOSS_GUIDE.md`** - Complete guide (READ THIS!)
5. **`energy_regularization_old.py`** - Backup of broken version

## Key Takeaways

**Question**: How do you make sure energy loss is in line with MSE?

**Answer**:
1. ✓ **Normalize by statistics**: Divide by std² for 'relative' mode
2. ✓ **Scale to [0,1] range**: Use normalized mode
3. ✓ **Compute from training data**: Automatically calibrate
4. ✓ **Monitor during training**: Check that both losses contribute
5. ✓ **Tune weights**: Start 0.8/0.2, adjust based on results

**Question**: How do you correctly punish OOD and reward in-distribution?

**Answer**:
1. ✓ **'relative' mode**: Gaussian penalty - increases with distance from mean
2. ✓ **'threshold' mode**: Zero penalty within 2σ, linear beyond
3. ✓ **Both properly scaled**: Comparable to MSE

## Next Steps

1. **Read**: [`ENERGY_LOSS_GUIDE.md`](ENERGY_LOSS_GUIDE.md) for detailed math
2. **Train**: Use `./run_fno_training.sh` with fixed version
3. **Monitor**: Check that losses are balanced during training
4. **Tune**: Adjust weights if needed

The bug is fixed and everything is properly scaled now! 🎉
