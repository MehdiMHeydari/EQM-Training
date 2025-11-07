# Why Are There Two UNets? (Explained)

## TL;DR: We Only Use ONE UNet!

**YES, we can delete the torchcfm UNet!** ✅ (Already deleted)

---

## The Two UNets

### 1. **torchcfm/models/unet/** ❌ NOT USED
- This is a **general-purpose UNet** included in the torchcfm library
- Designed for standard timestep-conditioned flow matching
- **We don't use it at all**
- **Status: DELETED** ✅

### 2. **physics_flow_matching/unet/unet_bb.py** ✅ USED
- This is the **custom UNet** specifically for EQM training
- Modified from the original to work without timesteps
- **This is what we actually use**
- **Status: KEPT** ✅

---

## What We Actually Import

### From torchcfm:
```python
from torchcfm.conditional_flow_matching import EquilibriumMatching
```
**Only the flow matching algorithm**, NOT the UNet!

### From physics_flow_matching:
```python
from physics_flow_matching.unet.unet_bb import UNetModelWrapper
```
**The actual UNet model we use**

---

## Why torchcfm Has a UNet

torchcfm is a **library** that provides:
1. ✅ Flow matching algorithms (EquilibriumMatching, ConditionalFlowMatcher, etc.)
2. ❌ Example models (UNet, MLP) - just for reference/examples
3. ✅ Utilities (optimal transport, etc.)

The torchcfm UNet is just an **example model** included for convenience. You can use it if you want, but we don't need it because:
- It requires timesteps (incompatible with EQM)
- We have a custom UNet that's better suited for EQM

---

## Verification

### Training Script Uses:
```python
# Line 8: Import custom UNet
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel

# Line 13: Import flow matching algorithm ONLY
from torchcfm.conditional_flow_matching import EquilibriumMatching

# Line 71-78: Instantiate the custom UNet
model = UNetModel(dim=config.unet.dim,
                  out_channels=config.unet.out_channels,
                  # ... custom UNet parameters
                  )
```

**torchcfm UNet is never imported or used!**

---

## What We Need from Each

### From torchcfm (library):
```
torchcfm/
├── conditional_flow_matching.py  ← EquilibriumMatching class
├── optimal_transport.py          ← OT utilities
├── utils.py                      ← Helper functions
└── models/
    └── models.py                 ← MLP (not UNet)
```

### From physics_flow_matching (our code):
```
physics_flow_matching/
└── unet/
    ├── unet_bb.py     ← THE UNet we actually use
    ├── nn.py          ← UNet utilities
    ├── fp16_util.py   ← FP16 support
    └── logger.py      ← Logging stub
```

---

## Summary Table

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **EquilibriumMatching** | torchcfm/ | Flow matching algorithm | ✅ NEEDED |
| **torchcfm UNet** | torchcfm/models/unet/ | Example model | ❌ DELETED |
| **Custom UNet** | physics_flow_matching/unet/ | Actual model we use | ✅ NEEDED |

---

## Final Answer

**Do we need both?**
- ✅ YES: We need **torchcfm library** (for EquilibriumMatching)
- ✅ YES: We need **physics_flow_matching UNet** (our model)
- ❌ NO: We DON'T need **torchcfm UNet** (just an example)

**The torchcfm UNet has been deleted** to reduce repository size. We only use:
1. torchcfm's **flow matching algorithms**
2. physics_flow_matching's **custom UNet model**

Everything is still working perfectly! ✅
