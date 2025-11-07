# EQM Training for Darcy Flow - Google Colab Ready! 🚀

Streamlined repository for training Equilibrium Matching models on Darcy Flow HDF5 datasets.

## 🎯 Quick Start for Google Colab

### Option 1: Use the Ready-Made Notebook (Easiest!)

1. **Upload your data** to Google Drive:
   - File: `2D_DarcyFlow_beta1.0_Train.hdf5`

2. **Open in Colab**:
   - Upload `EQM_Darcy_Training.ipynb` to Google Colab
   - Or open directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MehdiMHeydari/EQM-Training/blob/main/EQM_Darcy_Training.ipynb)

3. **Enable GPU**:
   - Runtime → Change runtime type → GPU

4. **Update data path**:
   - In Cell 3, change `DRIVE_DATA_PATH` to your file location

5. **Run all cells**:
   - Runtime → Run all
   - Training starts automatically! ☕

### Option 2: Manual Setup

```python
# 1. Clone repository
!git clone https://github.com/MehdiMHeydari/EQM-Training.git
%cd EQM-Training

# 2. Mount Drive and copy data
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/2D_DarcyFlow_beta1.0_Train.hdf5 data/

# 3. Install dependencies
!pip install torch torchvision h5py einops omegaconf tensorboard POT
!pip install -e .

# 4. Run training
!python physics_flow_matching/train_scripts/train_unet_eqm.py configs/darcy_flow_eqm.yaml
```

---

## 📚 Documentation

- **[COLAB_SETUP.md](COLAB_SETUP.md)**: Detailed Colab setup guide
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)**: What was removed and why
- **[TWO_UNETS_EXPLAINED.md](TWO_UNETS_EXPLAINED.md)**: UNet architecture explanation

---

## 🏗️ Repository Structure

```
EQM-Training/
├── EQM_Darcy_Training.ipynb       # Ready-to-use Colab notebook
├── COLAB_SETUP.md                 # Detailed Colab guide
├── configs/
│   └── darcy_flow_eqm.yaml        # Training configuration
├── physics_flow_matching/
│   ├── train_scripts/
│   │   └── train_unet_eqm.py      # Main training script
│   ├── utils/
│   │   ├── dataset.py             # DarcyFlow HDF5 dataset
│   │   ├── train_eqm.py           # Training loop
│   │   ├── pre_procs_data.py      # Data preprocessing
│   │   └── obj_funcs.py           # Loss functions
│   └── unet/
│       ├── unet_bb.py             # Custom UNet for EQM
│       ├── nn.py                  # NN utilities
│       └── fp16_util.py           # FP16 support
└── torchcfm/                      # Flow matching library
```

---

## ⚙️ Configuration

Edit `configs/darcy_flow_eqm.yaml` or modify in notebook:

```yaml
device: cuda                # GPU device
num_epochs: 100            # Training epochs
dataloader:
  batch_size: 32           # Batch size (reduce if OOM)
  dataset: DarcyFlow       # Dataset type
unet:
  num_channels: 64         # Model capacity
optimizer:
  lr: 0.0001              # Learning rate
```

---

## 🔍 What This Does

**Equilibrium Matching (EQM)** trains a neural network to learn the data distribution by:
1. Learning an energy function `E(x)`
2. Computing velocity field via `v(x) = ∇E(x)`
3. Sampling from learned distribution using ODE solver

**Your Darcy Flow Data**:
- Input: Permeability field `ν(x)` (10,000 samples, 128×128)
- Output: Solution field `u(x)` (10,000 samples, 128×128)

The model learns to generate realistic Darcy flow solutions!

---

## 💾 Saving Results

### To Google Drive (Recommended)
```python
import shutil
shutil.copytree("experiments/darcy_flow_eqm",
                "/content/drive/MyDrive/EQM_Experiments")
```

### Download Directly
```python
from google.colab import files
!zip -r results.zip experiments/darcy_flow_eqm
files.download('results.zip')
```

---

## 📊 Monitoring Training

### TensorBoard (in Colab)
```python
%load_ext tensorboard
%tensorboard --logdir experiments/darcy_flow_eqm
```

### Checkpoints
Saved every 10 epochs to:
```
experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_*.pth
```

---

## 🐛 Common Issues

### "CUDA out of memory"
**Solution**: Reduce batch size
```python
config.dataloader.batch_size = 16  # or 8, or 4
```

### "Runtime disconnected"
**Solution**: Resume training
```python
config.restart = True
config.restart_epoch = 40  # last completed epoch
```

### "No GPU detected"
**Solution**: Enable GPU
- Runtime → Change runtime type → Hardware accelerator → GPU

### More help
See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed troubleshooting

---

## 🎓 Citation

This repository is based on:
- **Conditional Flow Matching**: [Original Repo](https://github.com/atong01/conditional-flow-matching)
- **Equilibrium Matching**: [Paper](https://arxiv.org/abs/2406.04375)

---

## 📝 License

Same as original repository (see LICENSE file)

---

## ✨ What's Different from Original?

This is a **streamlined version** (~70-80% smaller) focused on:
- ✅ EQM training only
- ✅ HDF5 data support (Darcy Flow)
- ✅ Google Colab ready
- ❌ Removed: examples, tests, other training methods

For the full repository, see: [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)

---

## 🚀 Ready to Train!

1. Open `EQM_Darcy_Training.ipynb` in Colab
2. Upload your HDF5 file to Drive
3. Run all cells
4. Watch the magic happen! ✨

Questions? Check [COLAB_SETUP.md](COLAB_SETUP.md) for detailed instructions!
