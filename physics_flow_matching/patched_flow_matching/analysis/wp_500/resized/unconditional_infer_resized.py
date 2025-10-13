import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.uncond import infer_patched
import torch.nn.functional as F

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
m = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/m_500.npy")
std_old = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/std_500.npy")

# %%
exp = "2_limited_data"
iteration = 199
model = UNetModel(dim=[3, 32, 32],
                        channel_mult=(1,2,2),
                        num_channels=128,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="16, 8",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        class_cond=False,
                        num_classes=None,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/exps_patch_no_inner_scaling_coord/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
model.load_state_dict(state["model_state_dict"])
model.to(device)
model.eval();

# %%
samples = infer_patched(dims_of_img=(1,599,399), patch_size=(56,56), total_samples=500, samples_per_batch=10, cfm_model=model, t_start=0, t_end=1, scale=False, t_steps=100, device=device, ignore_index=1)

# %%
samples_ = samples*std_old + m

np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/exps_patch_no_inner_scaling_coord/exp_{exp}/samples_{iteration}.npy", samples_)