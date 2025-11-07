# Google Colab Setup Guide for EQM Darcy Flow Training

## Quick Start (TL;DR)

1. Upload your HDF5 file to Google Drive
2. Open the `EQM_Darcy_Training.ipynb` notebook in Colab
3. Run all cells
4. Training starts automatically!

---

## Detailed Setup Instructions

### Step 1: Prepare Your Data

1. Upload `2D_DarcyFlow_beta1.0_Train.hdf5` to your Google Drive
2. Note the path (e.g., `/content/drive/MyDrive/2D_DarcyFlow_beta1.0_Train.hdf5`)

### Step 2: Open Colab Notebook

Option A: **Use the provided notebook** (Recommended)
- Upload `EQM_Darcy_Training.ipynb` to Colab
- Open it and run all cells

Option B: **Create new notebook from scratch**
- Follow the cell-by-cell instructions below

---

## Colab Notebook Structure

### Cell 1: Enable GPU (Important!)

Go to: **Runtime → Change runtime type → Hardware accelerator → GPU**

### Cell 2: Clone Repository and Setup

```python
# Clone the repository
!git clone https://github.com/MehdiMHeydari/EQM-Training.git
%cd EQM-Training

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies (this may take 3-5 minutes)
!pip install -q torch torchvision
!pip install -q h5py einops omegaconf tensorboard POT
!pip install -q -e .

print("✅ Setup complete!")
```

### Cell 3: Copy Data from Drive

```python
import os
import shutil

# CHANGE THIS PATH to match your Google Drive location
drive_data_path = "/content/drive/MyDrive/2D_DarcyFlow_beta1.0_Train.hdf5"

# Copy to local data folder
local_data_path = "data/2D_DarcyFlow_beta1.0_Train.hdf5"

if os.path.exists(drive_data_path):
    print(f"Copying data from Drive...")
    os.makedirs("data", exist_ok=True)
    shutil.copy(drive_data_path, local_data_path)
    print(f"✅ Data copied successfully!")

    # Verify file size
    size_mb = os.path.getsize(local_data_path) / (1024**2)
    print(f"   File size: {size_mb:.2f} MB")
else:
    print(f"❌ File not found at: {drive_data_path}")
    print("Please update the drive_data_path variable!")
```

### Cell 4: Verify Installation

```python
# Quick test
from physics_flow_matching.utils.dataset import DarcyFlow
from physics_flow_matching.unet.unet_bb import UNetModelWrapper
from torchcfm.conditional_flow_matching import EquilibriumMatching

print("Testing dataset loading...")
dataset = DarcyFlow(
    hdf5_path="data/2D_DarcyFlow_beta1.0_Train.hdf5",
    normalize=True,
    use_eqm_format=True
)

print(f"✅ Dataset loaded: {len(dataset)} samples")
print(f"   Sample shape: {dataset.shape}")
print("\n🎉 Everything is ready for training!")
```

### Cell 5: Configure Training (Optional)

```python
# View current config
!cat configs/darcy_flow_eqm.yaml

# Or modify config programmatically
from omegaconf import OmegaConf

config = OmegaConf.load("configs/darcy_flow_eqm.yaml")

# Adjust settings for Colab
config.device = "cuda"  # Use GPU
config.dataloader.batch_size = 16  # Reduce if OOM
config.num_epochs = 50  # Adjust as needed
config.path = "./experiments/darcy_flow_eqm"

# Save modified config
OmegaConf.save(config, "configs/darcy_flow_eqm_colab.yaml")
print("✅ Config updated for Colab!")
```

### Cell 6: Start Training

```python
# Start training
!python physics_flow_matching/train_scripts/train_unet_eqm.py configs/darcy_flow_eqm.yaml
```

### Cell 7: Monitor Training (Optional - Run in parallel)

```python
# Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir experiments/darcy_flow_eqm
```

### Cell 8: Save Model to Drive (After Training)

```python
import shutil

# Copy trained model back to Drive
experiment_path = "experiments/darcy_flow_eqm"
drive_save_path = "/content/drive/MyDrive/EQM_Experiments"

if os.path.exists(experiment_path):
    print("Saving experiment to Google Drive...")
    shutil.copytree(experiment_path, drive_save_path, dirs_exist_ok=True)
    print(f"✅ Experiment saved to: {drive_save_path}")
else:
    print("No experiment found!")
```

---

## Tips for Colab

### Memory Management

If you get **Out of Memory (OOM)** errors:

1. **Reduce batch size**:
   ```yaml
   dataloader:
     batch_size: 8  # Try 8, 4, or even 2
   ```

2. **Reduce model size**:
   ```yaml
   unet:
     num_channels: 32  # Down from 64
   ```

3. **Clear memory between runs**:
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```

### Runtime Limits

**Free Colab**:
- 12 hour maximum runtime
- GPU disconnects after ~90 min idle
- Save checkpoints regularly!

**Colab Pro**:
- 24 hour runtime
- Better GPUs
- Fewer disconnections

### Auto-Save Checkpoints

The training script already saves checkpoints every 10 epochs to:
```
experiments/darcy_flow_eqm/exp_1/saved_state/
```

To resume training after disconnect:
```yaml
restart: True
restart_epoch: 40  # Last completed epoch
```

---

## Common Issues and Solutions

### Issue 1: "No module named 'torch'"
**Solution**: Make sure Cell 2 ran successfully. Re-run it.

### Issue 2: "CUDA out of memory"
**Solution**: Reduce batch_size in config (see Memory Management above)

### Issue 3: "File not found: 2D_DarcyFlow_beta1.0_Train.hdf5"
**Solution**: Update `drive_data_path` in Cell 3 to match your Drive location

### Issue 4: Runtime disconnected
**Solution**:
- Copy `experiments/` folder to Drive (Cell 8)
- Resume training with `restart: True` in config
- Consider Colab Pro for longer runtimes

### Issue 5: Training is slow
**Solution**:
- Verify GPU is enabled (Runtime → Change runtime type)
- Check GPU usage: `!nvidia-smi`
- Increase batch_size if GPU memory allows

---

## Advanced: Custom Configuration

Create a custom config for your needs:

```python
from omegaconf import OmegaConf

config = OmegaConf.create({
    "device": "cuda",
    "exp_num": 1,
    "path": "./experiments/darcy_flow_custom",
    "th_seed": 42,
    "np_seed": 42,

    "unet": {
        "dim": [1, 128, 128],
        "out_channels": 1,
        "channel_mult": "1, 1, 2, 3, 4",
        "num_channels": 64,
        "res_blocks": 2,
        "head_chans": 64,
        "attn_res": "32, 16, 8",
        "dropout": 0.1,  # Add dropout
        "new_attn": True,
        "film": True,
        "class_cond": False,
    },

    "FM": {
        "sched": "trunc_decay",
        "lamda": 4.0,
        "return_noise": False,
    },

    "dataloader": {
        "datapath": "data/2D_DarcyFlow_beta1.0_Train.hdf5",
        "batch_size": 32,
        "dataset": "DarcyFlow",
        "contrastive": False,
        "input_key": "nu",
        "output_key": "tensor",
        "normalize": True,
        "use_eqm_format": True,
    },

    "optimizer": {
        "lr": 1e-4,
    },

    "num_epochs": 100,
    "print_epoch_int": 1,
    "save_epoch_int": 10,
    "print_with_epoch_int": 50,
    "restart": False,
    "restart_epoch": None,
})

OmegaConf.save(config, "configs/my_custom_config.yaml")
```

---

## Quick Checklist

Before running training:
- [ ] GPU enabled in Colab
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Google Drive mounted
- [ ] HDF5 file copied to `data/`
- [ ] Config file ready
- [ ] Test imports passed

---

## Expected Training Time

On Colab GPU (T4):
- **Per epoch**: ~2-3 minutes (batch_size=32)
- **100 epochs**: ~3-5 hours

On Colab Pro GPU (V100/A100):
- **Per epoch**: ~1-2 minutes
- **100 epochs**: ~2-3 hours

---

## Next Steps After Training

1. **Download checkpoints**:
   ```python
   from google.colab import files
   !zip -r checkpoints.zip experiments/darcy_flow_eqm/exp_1/saved_state/
   files.download('checkpoints.zip')
   ```

2. **Visualize results**:
   - Use TensorBoard (Cell 7)
   - Plot loss curves
   - Generate samples

3. **Fine-tune**:
   - Adjust learning rate
   - Try different architectures
   - Experiment with hyperparameters

---

## Support

If you encounter issues:
1. Check this guide's "Common Issues" section
2. Review the error message carefully
3. Check TensorBoard for training metrics
4. Verify GPU memory usage: `!nvidia-smi`

Happy training! 🚀
