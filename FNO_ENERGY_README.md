# FNO Training with Energy Regularization

Train a Fourier Neural Operator (FNO) on Darcy flow with energy-based regularization to keep predictions physically plausible and in-distribution.

## Overview

This integrates your trained EQM (Equilibrium Matching) energy function as a regularization loss for neural operator training:

```
Total Loss = 0.8 × MSE(prediction, target) + 0.2 × Energy(prediction)
```

**Why this matters:**
- **MSE alone**: Model overfits to training data, can produce out-of-distribution predictions
- **MSE + Energy**: Model learns to stay in-distribution, produces physically plausible solutions
- **Energy regularization**: Penalizes predictions with high energy (OOD), encourages low energy (in-distribution)

## Quick Start

### Local Machine

```bash
# Make script executable
chmod +x run_fno_training.sh

# Run training
./run_fno_training.sh
```

### Google Colab

```python
# In a Colab cell:
!python /content/EQM-Training/FNO_Energy_Colab_Setup.py
```

Then follow the printed instructions to start training.

## What Each File Does

### Core Files

1. **`energy_regularization.py`** (already created)
   - `EnergyRegularizationLoss`: Computes energy E(x) = sum(x * model(x)) using trained EQM model
   - `CombinedLoss`: Combines MSE + Energy with configurable weights
   - Handles normalization automatically

2. **`train_fno_with_energy.py`** (NEW)
   - Complete training script for FNO with energy regularization
   - Loads Darcy flow data and normalizes to [-1, 1]
   - Integrates with Neural-Solver-Library's FNO implementation
   - Tracks MSE, energy, and energy statistics during training
   - Saves checkpoints and plots training curves

3. **`run_fno_training.sh`**
   - Convenient training script with default hyperparameters
   - Edit this to change batch size, learning rate, loss weights, etc.

4. **`FNO_Energy_Colab_Setup.py`**
   - One-command setup for Google Colab
   - Clones repos, installs dependencies, mounts Drive
   - Verifies checkpoints and data paths

## Training Configuration

### Key Hyperparameters

```python
# Loss weights (tune these!)
--mse_weight 0.8          # Weight for MSE loss (data fitting)
--energy_weight 0.2       # Weight for energy regularization (physics-informed)
--energy_temperature 1.0  # Temperature for energy scaling

# Training
--batch_size 4            # Batch size (adjust based on GPU memory)
--epochs 100              # Number of training epochs
--lr 1e-3                 # Learning rate

# Data
--train_samples 800       # Number of training samples
--val_samples 200         # Number of validation samples
```

### Loss Weight Recommendations

**Start with 0.8/0.2 split**, then experiment:

| MSE Weight | Energy Weight | Use Case |
|------------|---------------|----------|
| 0.9 | 0.1 | More emphasis on data fitting |
| **0.8** | **0.2** | **Balanced (recommended start)** |
| 0.7 | 0.3 | More emphasis on staying in-distribution |
| 0.5 | 0.5 | Equal importance (may hurt MSE) |

**How to tune:**
- If **MSE is high**: Increase MSE weight, decrease energy weight
- If **predictions look unphysical**: Increase energy weight, decrease MSE weight
- If **energy is increasing during training**: Model is drifting OOD, increase energy weight

## Expected Results

### Training Progress

You should see:

```
Epoch 1/100:
  Train: loss=0.0850 (mse=0.0920, energy=-52341.23)
  Val:   loss=0.0823 (mse=0.0891, energy=-52156.78)
  Energy stats: μ=-52156.8, σ=3421.5

Epoch 50/100:
  Train: loss=0.0234 (mse=0.0198, energy=-49823.45)
  Val:   loss=0.0241 (mse=0.0205, energy=-49654.12)
  Energy stats: μ=-49654.1, σ=3102.8
  → Best model saved! (val_loss=0.0241)
```

**Good signs:**
- ✓ MSE decreases over epochs
- ✓ Energy stays relatively stable or decreases slightly
- ✓ Energy std is reasonably wide (2000-4000)
- ✓ Val loss tracks train loss (not overfitting)

**Bad signs:**
- ✗ Energy increases dramatically → predictions going OOD
- ✗ Energy std becomes very narrow (<100) → mode collapse
- ✗ MSE plateaus early → increase learning rate or train longer

### Output Files

After training, you'll have:

1. **`fno_with_energy.pth`** - Final model checkpoint
2. **`fno_with_energy_best.pth`** - Best validation loss checkpoint
3. **`fno_with_energy_epoch25.pth`**, etc. - Intermediate checkpoints
4. **`fno_training_curves.png`** - Training curves showing:
   - Total loss (MSE + Energy)
   - MSE loss (data fitting)
   - Energy regularization loss
   - Energy statistics over training

## FNO Architecture Details

**Why FNO for Darcy Flow:**

1. **Spectral Operator**: FNO learns in Fourier space
   - Captures global patterns efficiently
   - Natural for smooth PDE solutions like Darcy flow

2. **Grid-Based**:
   - Input: 128×128 permeability field
   - Output: 128×128 pressure/solution field
   - Perfectly compatible with EQM energy model

3. **4 Fourier Layers**:
   - Each layer mixes spatial and spectral information
   - 12 Fourier modes per layer (controllable)
   - 128 hidden dimensions

4. **Total Parameters**: ~1.5M (fast to train, easy to regularize)

## Comparison: FNO vs FNO+Energy

Once training finishes, you can compare:

### Without Energy Regularization (baseline FNO)
```bash
# Train baseline FNO (set energy_weight=0)
python train_fno_with_energy.py \
    --energy_weight 0.0 \
    --mse_weight 1.0 \
    --checkpoint_save_path fno_baseline.pth
```

### With Energy Regularization
```bash
# Already done with default script!
```

**Expected differences:**
- Baseline FNO: Lower MSE, but may produce OOD predictions
- FNO + Energy: Slightly higher MSE, but predictions stay in-distribution
- FNO + Energy: Better generalization to unseen boundary conditions

## Monitoring Training

### Watch Training in Real-Time

```bash
# If running locally, tail the output
python train_fno_with_energy.py ... | tee training.log
```

### Check Validation Energy

After each epoch, check:
- **Energy mean**: Should be close to ground truth energy (~-51,733)
- **Energy std**: Should be reasonably wide (~2000-4000)

If energy mean is much lower (more negative) than ground truth:
- Model is over-optimizing to spurious low-energy states
- Increase energy weight or reduce learning rate

## Troubleshooting

### GPU Out of Memory

Reduce batch size:
```bash
--batch_size 2  # or even 1
```

### Training Too Slow

- Use smaller model (reduce `n_hidden` in `Args` class)
- Reduce number of epochs
- Use mixed precision training (requires code modification)

### Energy Increasing During Training

This means predictions are drifting OOD. Solutions:
1. Increase energy weight: `--energy_weight 0.3`
2. Reduce learning rate: `--lr 5e-4`
3. Add gradient clipping: `--max_grad_norm 1.0`

### MSE Not Decreasing

1. Check learning rate (may be too low)
2. Ensure data is loaded correctly (check normalization)
3. Try reducing energy weight temporarily: `--energy_weight 0.1`

## Advanced: Modifying the Loss

### Change Loss Weights During Training

Edit `train_fno_with_energy.py`:

```python
# Around line where criterion is called
if epoch < 50:
    # First 50 epochs: focus on MSE
    criterion.mse_weight = 0.9
    criterion.energy_weight = 0.1
else:
    # After 50 epochs: increase energy regularization
    criterion.mse_weight = 0.7
    criterion.energy_weight = 0.3
```

### Add Energy Statistics Logging

Already included! Check `history` dict for:
- `energy_mean`: Mean energy of predictions per epoch
- `energy_std`: Std of energy per epoch
- Plotted automatically in training curves

## Next Steps

After training:

1. **Evaluate on Test Set**
   - Compare MSE with/without energy regularization
   - Check energy distribution of predictions

2. **Visualize Predictions**
   - Plot sample predictions vs ground truth
   - Verify they look physically plausible

3. **Try Other Neural Operators**
   - U-FNO: Better for multi-scale features
   - Transolver: State-of-the-art transformer-based
   - Just change `--model` argument (requires modifying code)

4. **Tune Loss Weights**
   - Try 0.7/0.3, 0.9/0.1, 0.5/0.5 splits
   - Find optimal balance for your use case

5. **Apply to Other PDEs**
   - Navier-Stokes, wave equation, etc.
   - Just change data loader and config

## Citation

If you use this in your research, please cite:

**FNO:**
```bibtex
@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={ICLR},
  year={2021}
}
```

**Neural-Solver-Library:**
```bibtex
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={ICML},
  year={2024}
}
```

## Contact

Issues? Questions?
- Check GitHub issues in both repos
- EQM Training: https://github.com/MehdiMHeydari/EQM-Training
- Neural-Solver-Library: https://github.com/thuml/Neural-Solver-Library
