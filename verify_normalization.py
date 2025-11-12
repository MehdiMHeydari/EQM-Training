"""
Quick script to verify min-max normalization is working correctly.
Run this before training to ensure data is properly normalized to [-1, 1].
"""

import numpy as np
import h5py
from physics_flow_matching.utils.dataset import DATASETS

# Load dataset with normalization
print("=" * 60)
print("TESTING MIN-MAX NORMALIZATION")
print("=" * 60)

dataset = DATASETS["DarcyFlow"](
    hdf5_path="data/2D_DarcyFlow_beta1.0_Train.hdf5",
    input_key="nu",
    output_key="tensor",
    normalize=True,
    use_eqm_format=True
)

# Get normalization stats
norm_stats = dataset.get_normalization_stats()

print("\n1. NORMALIZATION STATISTICS (from dataset):")
print(f"   Output data_min: {norm_stats['data_min']:.6f}")
print(f"   Output data_max: {norm_stats['data_max']:.6f}")
print(f"   Input input_min: {norm_stats['input_min']:.6f}")
print(f"   Input input_max: {norm_stats['input_max']:.6f}")

# Check actual normalized data range
print("\n2. ACTUAL NORMALIZED DATA RANGE:")
output_min = dataset.output_data.min()
output_max = dataset.output_data.max()
output_mean = dataset.output_data.mean()

print(f"   Output min: {output_min:.6f}")
print(f"   Output max: {output_max:.6f}")
print(f"   Output mean: {output_mean:.6f}")

input_min = dataset.input_data.min()
input_max = dataset.input_data.max()
input_mean = dataset.input_data.mean()

print(f"   Input min: {input_min:.6f}")
print(f"   Input max: {input_max:.6f}")
print(f"   Input mean: {input_mean:.6f}")

# Verify it's in [-1, 1] range
print("\n3. VERIFICATION:")
if -1.0 <= output_min <= output_max <= 1.0:
    print("   ✓ Output data is correctly normalized to [-1, 1]")
else:
    print(f"   ✗ WARNING: Output data is NOT in [-1, 1] range!")

if -1.0 <= input_min <= input_max <= 1.0:
    print("   ✓ Input data is correctly normalized to [-1, 1]")
else:
    print(f"   ✗ WARNING: Input data is NOT in [-1, 1] range!")

# Load original data to show before/after
print("\n4. ORIGINAL DATA RANGE (before normalization):")
with h5py.File("data/2D_DarcyFlow_beta1.0_Train.hdf5", 'r') as f:
    original_output = np.array(f['tensor']).astype(np.float32)
    original_input = np.array(f['nu']).astype(np.float32)

print(f"   Original output min: {original_output.min():.6f}")
print(f"   Original output max: {original_output.max():.6f}")
print(f"   Original input min: {original_input.min():.6f}")
print(f"   Original input max: {original_input.max():.6f}")

# Test a few samples
print("\n5. SAMPLE TEST (first 3 datapoints):")
for i in range(min(3, len(dataset))):
    x0, x1 = dataset[i]
    print(f"   Sample {i}: x0 range=[{x0.min():.3f}, {x0.max():.3f}], x1 range=[{x1.min():.3f}, {x1.max():.3f}]")

print("\n6. DATASET INFO:")
print(f"   Total samples: {len(dataset)}")
print(f"   Sample shape: {dataset.shape}")
print(f"   EQM format: {dataset.use_eqm_format}")

print("\n" + "=" * 60)
print("NORMALIZATION CHECK COMPLETE!")
print("=" * 60)

# Final summary
print("\nSUMMARY:")
if -1.0 <= output_min <= output_max <= 1.0 and -1.0 <= input_min <= input_max <= 1.0:
    print("✓ All data is properly normalized to [-1, 1]")
    print("✓ Safe to proceed with training")
else:
    print("✗ NORMALIZATION ERROR - DO NOT TRAIN YET!")
    print("  Check the normalization code in dataset.py")
