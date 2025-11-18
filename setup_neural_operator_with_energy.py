"""
Setup script for Google Colab: Neural Operator training with Energy Regularization

This script:
1. Clones Neural-Solver-Library
2. Clones your EQM-Training repo
3. Integrates energy-based regularization loss
4. Sets up training with 0.8*MSE + 0.2*Energy loss

Run this in Google Colab!
"""

import os
import sys

print("="*70)
print("SETUP: Neural Operator with Energy Regularization")
print("="*70)

# Step 1: Clone repositories
print("\n[1/5] Cloning repositories...")

# Clone Neural-Solver-Library
if not os.path.exists('Neural-Solver-Library'):
    print("Cloning Neural-Solver-Library...")
    !git clone https://github.com/thuml/Neural-Solver-Library.git
    print("✓ Neural-Solver-Library cloned")
else:
    print("✓ Neural-Solver-Library already exists")

# Clone your EQM repo
if not os.path.exists('EQM-Training'):
    print("\nCloning EQM-Training...")
    !git clone https://github.com/MehdiMHeydari/EQM-Training.git
    print("✓ EQM-Training cloned")
else:
    print("✓ EQM-Training already exists")
    !cd EQM-Training && git pull

# Step 2: Install dependencies
print("\n[2/5] Installing dependencies...")
!pip install -q torch torchvision einops omegaconf h5py matplotlib scipy tqdm

# Add paths to sys.path
sys.path.append('/content/EQM-Training')
sys.path.append('/content/Neural-Solver-Library')

print("✓ Dependencies installed")

# Step 3: Mount Google Drive
print("\n[3/5] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted")

# Step 4: Verify EQM checkpoint exists
print("\n[4/5] Verifying EQM checkpoint...")
checkpoint_path = "/content/drive/MyDrive/EQM_Checkpoints5/checkpoint_90.pth"

if os.path.exists(checkpoint_path):
    print(f"✓ Found EQM checkpoint: {checkpoint_path}")
else:
    print(f"✗ WARNING: Checkpoint not found at {checkpoint_path}")
    print("  Please update the checkpoint_path variable")

# Step 5: Create training script
print("\n[5/5] Creating neural operator training script with energy regularization...")

training_script = '''
"""
Neural Operator Training with Energy Regularization

Train a neural operator (FNO, U-Net, etc.) on Darcy flow with:
- 80% MSE loss (data fitting)
- 20% Energy regularization (physics-informed, keeps predictions in-distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add paths
sys.path.append('/content/EQM-Training')
sys.path.append('/content/Neural-Solver-Library')

from energy_regularization import CombinedLoss

# Configuration
CONFIG = {
    # Paths
    'checkpoint_path': '/content/drive/MyDrive/EQM_Checkpoints5/checkpoint_90.pth',
    'config_path': '/content/EQM-Training/configs/darcy_flow_eqm.yaml',
    'data_path': '/content/EQM-Training/data/2D_DarcyFlow_beta1.0_Train.hdf5',

    # Loss weights
    'mse_weight': 0.8,
    'energy_weight': 0.2,
    'energy_temperature': 1.0,

    # Training
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Data
    'train_samples': 800,
    'val_samples': 200,
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# Simple U-Net for neural operator (replace with FNO or your preferred architecture)
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Middle
        m = self.middle(p2)

        # Decoder
        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return d1

# Load Darcy Flow data
print("Loading Darcy Flow data...")
with h5py.File(CONFIG['data_path'], 'r') as f:
    # For Darcy flow, we might have permeability 'a' as input and solution 'u' as output
    # Adjust keys based on your dataset
    if 'nu' in f.keys():
        inputs = np.array(f['nu']).astype(np.float32)[:, np.newaxis, :, :]
        outputs = np.array(f['tensor']).astype(np.float32)[:, np.newaxis, :, :]
    else:
        # If only 'tensor' exists, use it as both input and output for demo
        data = np.array(f['tensor']).astype(np.float32)[:, np.newaxis, :, :]
        inputs = data
        outputs = data
        print("  Note: Using same data for input/output (demo mode)")

# Split train/val
train_inputs = torch.FloatTensor(inputs[:CONFIG['train_samples']])
train_outputs = torch.FloatTensor(outputs[:CONFIG['train_samples']])
val_inputs = torch.FloatTensor(inputs[CONFIG['train_samples']:CONFIG['train_samples']+CONFIG['val_samples']])
val_outputs = torch.FloatTensor(outputs[CONFIG['train_samples']:CONFIG['train_samples']+CONFIG['val_samples']])

train_dataset = TensorDataset(train_inputs, train_outputs)
val_dataset = TensorDataset(val_inputs, val_outputs)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Input shape: {train_inputs[0].shape}")
print(f"Output shape: {train_outputs[0].shape}")
print()

# Initialize model
device = torch.device(CONFIG['device'])
model = SimpleUNet(in_channels=1, out_channels=1).to(device)

# Initialize combined loss with energy regularization
print("Initializing combined loss (MSE + Energy regularization)...")
criterion = CombinedLoss(
    checkpoint_path=CONFIG['checkpoint_path'],
    config_path=CONFIG['config_path'],
    device=device,
    mse_weight=CONFIG['mse_weight'],
    energy_weight=CONFIG['energy_weight'],
    temperature=CONFIG['energy_temperature'],
    normalize_inputs=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
print()

# Training loop
print("Starting training...")
history = {
    'train_loss': [], 'train_mse': [], 'train_energy': [],
    'val_loss': [], 'val_mse': [], 'val_energy': [],
    'energy_mean': [], 'energy_std': []
}

for epoch in range(CONFIG['num_epochs']):
    # Training
    model.train()
    train_loss, train_mse, train_energy = 0, 0, 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)

        # Combined loss
        loss, loss_dict = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss_dict['total']
        train_mse += loss_dict['mse']
        train_energy += loss_dict['energy_reg']

    # Validation
    model.eval()
    val_loss, val_mse, val_energy = 0, 0, 0
    energy_means, energy_stds = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)

            loss, loss_dict = criterion(predictions, targets)
            energy_stats = criterion.get_energy_stats(predictions)

            val_loss += loss_dict['total']
            val_mse += loss_dict['mse']
            val_energy += loss_dict['energy_reg']
            energy_means.append(energy_stats['mean'])
            energy_stds.append(energy_stats['std'])

    # Average losses
    train_loss /= len(train_loader)
    train_mse /= len(train_loader)
    train_energy /= len(train_loader)
    val_loss /= len(val_loader)
    val_mse /= len(val_loader)
    val_energy /= len(val_loader)

    # Log
    history['train_loss'].append(train_loss)
    history['train_mse'].append(train_mse)
    history['train_energy'].append(train_energy)
    history['val_loss'].append(val_loss)
    history['val_mse'].append(val_mse)
    history['val_energy'].append(val_energy)
    history['energy_mean'].append(np.mean(energy_means))
    history['energy_std'].append(np.mean(energy_stds))

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} (MSE={train_mse:.4f}, Energy={train_energy:.4f}), "
          f"Val Loss={val_loss:.4f} (MSE={val_mse:.4f}, Energy={val_energy:.4f}), "
          f"Energy Stats: μ={history['energy_mean'][-1]:.1f}, σ={history['energy_std'][-1]:.1f}")

    scheduler.step(val_loss)

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Total loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Loss (0.8*MSE + 0.2*Energy)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MSE vs Energy
axes[1].plot(history['train_mse'], label='Train MSE')
axes[1].plot(history['val_mse'], label='Val MSE')
axes[1].plot(history['train_energy'], label='Train Energy', linestyle='--')
axes[1].plot(history['val_energy'], label='Val Energy', linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss Component')
axes[1].set_title('MSE vs Energy Regularization')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Energy statistics
axes[2].plot(history['energy_mean'], label='Mean Energy')
axes[2].fill_between(range(len(history['energy_mean'])),
                     np.array(history['energy_mean']) - np.array(history['energy_std']),
                     np.array(history['energy_mean']) + np.array(history['energy_std']),
                     alpha=0.3, label='±1 Std')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Energy E(x)')
axes[2].set_title('Prediction Energy Over Training\\n(Lower = More In-Distribution)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTraining complete!")
print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
print(f"Final energy: {history['energy_mean'][-1]:.1f} ± {history['energy_std'][-1]:.1f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'config': CONFIG
}, 'neural_operator_with_energy_reg.pth')
print("Model saved to: neural_operator_with_energy_reg.pth")
'''

# Write the training script
with open('/content/train_neural_operator_with_energy.py', 'w') as f:
    f.write(training_script)

print("✓ Training script created: /content/train_neural_operator_with_energy.py")

print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Run: python /content/train_neural_operator_with_energy.py")
print("2. Or load and customize the script in a new cell")
print("\nThe training will use:")
print("  - 80% MSE loss (data fitting)")
print("  - 20% Energy regularization (physics-informed)")
print("="*70)
