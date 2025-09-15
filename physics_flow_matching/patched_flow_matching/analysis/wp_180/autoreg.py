import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.cond import flow_padis
from physics_flow_matching.inference_scripts.uncond import infer_patched
from physics_flow_matching.inference_scripts.utils import inpainting2, grad_cost_func, sample_noise

# %%
# m, std = data.mean(axis=(0,2,3), keepdims=True), data.std(axis=(0,2,3), keepdims=True)
# m = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/20/m.npy")
# std = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/20/std.npy")

# data_ = (data - m)/std

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# one_gt_sample = torch.from_numpy(data_[-1:]).to(device)

# %%
exp = 1
iteration = 2
model = UNetModel(dim=[3, 32, 32],
                        channel_mult=(1,2,2),
                        num_channels=128,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="16, 8",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        class_cond=True,
                        num_classes=None,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
model.load_state_dict(state["model_state_dict"])
model.to(device)
model.eval();

# %%
total_samples = 200
stream_wise_length = 1024
window_size = 254

# %%
long_samples = []
for _ in range(total_samples):
    long_sample = []
    for i in range(stream_wise_length//window_size):
        if i == 0:
            samples = infer_patched(dims_of_img=(1,255,255), patch_size=(56,56), total_samples=1, samples_per_batch=1, cfm_model=model, t_start=0, t_end=1, scale=False, t_steps=200, device=device, ignore_index=1)
        else:
            measurements = torch.from_numpy(long_sample[-1][..., window_size:, :]).to(device)
            samples = flow_padis(FlowMatcher(1e-3), model, samples_per_batch=1, total_samples=1, dims_of_img=(1,255,255), patch_size=(56, 56), num_of_steps=200, grad_cost_func=grad_cost_func,
                            meas_func=inpainting2, conditioning=measurements, conditioning_scale=2., device=device, refine=1, sample_noise=sample_noise, use_heavy_noise=False, 
                            rf_start=False, swag=False, nu=3,  sx=0, ex=255-window_size, sy=0, ey=256, ignore_index=1)
        long_sample.append(samples)
    long = np.concat([gen if i==0 else  gen[..., 255-window_size:, :] for i, gen in enumerate(long_sample)], axis=2)
    long_samples.append(long)

# %%
long_samples_ = np.concat(long_samples, axis=0)

# %%
m = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/m.npy")
std = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/std.npy")
# data_ = (data - m)/std

# %%
np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/exp_{exp}/samples_uncond_autoreg_{iteration}.npy", long_samples_*std + m)



