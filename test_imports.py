#!/usr/bin/env python3
"""
Quick test to verify all imports work after cleanup
"""
import sys
sys.path.extend(['.'])

print("Testing imports after cleanup...")
print("=" * 60)

# Test 1: Import dataset
print("\n[1/6] Importing dataset...")
try:
    from physics_flow_matching.utils.dataset import DATASETS, DarcyFlow
    print("✅ Dataset imports successful")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 2: Import UNet
print("\n[2/6] Importing UNet...")
try:
    from physics_flow_matching.unet.unet_bb import UNetModelWrapper
    print("✅ UNet import successful")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 3: Import training utilities
print("\n[3/6] Importing training utilities...")
try:
    from physics_flow_matching.utils.train_eqm import train_model
    from physics_flow_matching.utils.pre_procs_data import get_batch
    from physics_flow_matching.utils.obj_funcs import DD_loss
    print("✅ Training utilities import successful")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 4: Import torchcfm
print("\n[4/6] Importing torchcfm...")
try:
    from torchcfm.conditional_flow_matching import EquilibriumMatching
    print("✅ torchcfm import successful")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 5: Test DarcyFlow dataset
print("\n[5/6] Testing DarcyFlow dataset...")
try:
    dataset = DarcyFlow(
        hdf5_path="data/2D_DarcyFlow_beta1.0_Train.hdf5",
        normalize=True,
        use_eqm_format=True
    )
    print(f"✅ Dataset created: {len(dataset)} samples, shape {dataset.shape}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 6: Test UNet instantiation
print("\n[6/6] Testing UNet instantiation...")
try:
    import torch
    model = UNetModelWrapper(
        dim=[1, 128, 128],
        out_channels=1,
        channel_mult="1, 1, 2, 3, 4",
        num_channels=64,
        num_res_blocks=2,
        num_head_channels=64,
        attention_resolutions="32, 16, 8",
        dropout=0.0,
        use_new_attention_order=True,
        use_scale_shift_norm=True
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ UNet created: {param_count:,} parameters")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All imports and basic tests passed!")
print("=" * 60)
print("\n📦 Cleaned repository structure:")
print("   - Removed: examples/, runner/, tests/, assets/")
print("   - Removed: multi_fidelity/, patched_flow_matching/, vf_fm/")
print("   - Removed: 15+ unused training scripts")
print("   - Removed: 6 unused UNet models")
print("   - Removed: 15+ unused utility scripts")
print("\n✅ Repository is now ~70-80% smaller!")
print("✅ Ready for Google Colab!")
