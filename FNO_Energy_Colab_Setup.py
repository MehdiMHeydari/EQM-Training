"""
Google Colab Setup for FNO Training with Energy Regularization

Run this script in Google Colab to set up everything needed for training
FNO with energy-based regularization.
"""

import os
import sys

print("="*70)
print("SETUP: FNO with Energy Regularization")
print("="*70)

# Step 1: Clone repositories
print("\n[1/6] Cloning repositories...")

if not os.path.exists('EQM-Training'):
    print("Cloning EQM-Training...")
    !git clone https://github.com/MehdiMHeydari/EQM-Training.git
    print("✓ EQM-Training cloned")
else:
    print("✓ EQM-Training already exists")
    !cd EQM-Training && git pull

if not os.path.exists('Neural-Solver-Library-main'):
    print("\nCloning Neural-Solver-Library...")
    !git clone https://github.com/thuml/Neural-Solver-Library.git
    !mv Neural-Solver-Library Neural-Solver-Library-main
    print("✓ Neural-Solver-Library cloned")
else:
    print("✓ Neural-Solver-Library already exists")

# Step 2: Install dependencies
print("\n[2/6] Installing dependencies...")
!pip install -q torch torchvision einops omegaconf h5py matplotlib scipy tqdm timm

# Add to Python path
sys.path.append('/content/EQM-Training')
sys.path.append('/content/Neural-Solver-Library-main')

print("✓ Dependencies installed")

# Step 3: Mount Google Drive
print("\n[3/6] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted")

# Step 4: Verify EQM checkpoint
print("\n[4/6] Verifying EQM checkpoint...")
checkpoint_path = "/content/drive/MyDrive/EQM_Checkpoints5/checkpoint_90.pth"

if os.path.exists(checkpoint_path):
    print(f"✓ Found EQM checkpoint: {checkpoint_path}")
else:
    print(f"✗ WARNING: Checkpoint not found at {checkpoint_path}")
    print("  Please update the checkpoint_path variable below")
    checkpoint_path = input("Enter checkpoint path: ")

# Step 5: Verify data file
print("\n[5/6] Verifying data file...")
data_path = "/content/EQM-Training/data/2D_DarcyFlow_beta1.0_Train.hdf5"

if os.path.exists(data_path):
    print(f"✓ Found data file: {data_path}")
else:
    print(f"✗ WARNING: Data not found at {data_path}")
    print("  Please upload or specify data location")
    data_path = input("Enter data path: ")

# Step 6: Copy training script to Colab
print("\n[6/6] Setting up training script...")

# Check if energy_regularization.py exists
if os.path.exists('/content/EQM-Training/energy_regularization.py'):
    print("✓ energy_regularization.py found")
else:
    print("✗ WARNING: energy_regularization.py not found")
    print("  Make sure you've pulled the latest from GitHub")

if os.path.exists('/content/EQM-Training/train_fno_with_energy.py'):
    print("✓ train_fno_with_energy.py found")
else:
    print("✗ WARNING: train_fno_with_energy.py not found")
    print("  Make sure you've pulled the latest from GitHub")

print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)

print("\n" + "="*70)
print("TRAINING COMMAND")
print("="*70)
print("\nRun the following to start training:\n")

training_command = f"""
%cd /content/EQM-Training

!python train_fno_with_energy.py \\
    --data_path {data_path} \\
    --eqm_checkpoint {checkpoint_path} \\
    --eqm_config /content/EQM-Training/configs/darcy_flow_eqm.yaml \\
    --train_samples 800 \\
    --val_samples 200 \\
    --batch_size 4 \\
    --epochs 100 \\
    --lr 1e-3 \\
    --mse_weight 0.8 \\
    --energy_weight 0.2 \\
    --checkpoint_save_path /content/drive/MyDrive/fno_with_energy.pth \\
    --output_plot /content/drive/MyDrive/fno_training_curves.png \\
    --device cuda
"""

print(training_command)

print("\n" + "="*70)
print("WHAT THIS DOES:")
print("="*70)
print("""
1. Trains FNO (Fourier Neural Operator) on Darcy Flow
2. Uses combined loss: 0.8*MSE + 0.2*Energy
3. Energy regularization keeps predictions in-distribution
4. Prevents the model from producing physically implausible solutions
5. Saves best model and training curves to Google Drive

Expected training time: ~2-3 hours on Colab GPU
""")

print("="*70)

# Store paths for easy access
print("\n📍 Quick Reference:")
print(f"  EQM Checkpoint: {checkpoint_path}")
print(f"  Data Path: {data_path}")
print(f"  Config: /content/EQM-Training/configs/darcy_flow_eqm.yaml")
print("="*70)
