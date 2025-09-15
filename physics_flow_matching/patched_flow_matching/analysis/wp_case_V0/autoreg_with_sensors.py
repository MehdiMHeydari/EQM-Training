import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.cond import flow_padis_generalized
from physics_flow_matching.inference_scripts.uncond import infer_patched
from physics_flow_matching.inference_scripts.utils import inpainting2, grad_cost_func_generalized, sample_noise, inpainting
from einops import rearrange

# %%
m = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp_case_V0/m.npy")
std = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp_case_V0/std.npy")

# %%
data = np.load("/storage/yi/TBL_all_data/TBL_Re300/case_V0/case_v0_dt10_fluct.npy")[:, None]

# %%
data_ = (data - m)/std

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def meas_func_1(x, **kwargs):
    if 'start' not in kwargs.keys():
        return x * kwargs['mask']
    else:
        return x[..., kwargs['start']:, :] * kwargs['mask']

def meas_func_2(x, **kwargs):
    return x[..., kwargs['slice'], :]

total_samples = 1
samples_per_batch = 1
streamwise_length = 128
num_sensors = None

# %%
mask_ = np.zeros((1,1,512, 512))
mask_[..., ::22, ::22] = 1

# %%
conds = data_[::40] * mask_

# %%
exp = 1
iteration = 50
model = UNetModel(dim=[3, 32, 32],
                        channel_mult=(1,2,4,4),
                        num_channels=64,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="16, 8",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        class_cond=True,
                        num_classes=None,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp_case_V0/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
model.load_state_dict(state["model_state_dict"])
model.to(device)
model.eval();

# %%
total_samples = 1
window_length = 127
slic = slice(0, streamwise_length-window_length)

# %%
samples = []

for j, cond in enumerate(conds):
    print(f"Generating  sample : {j}")
    gen_sample = []
    for i in range(512//window_length):
        if i == 0:
            sample = flow_padis_generalized(fm = FlowMatcher(1e-3), cfm_model=model,
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(1,streamwise_length,512), num_of_steps=200, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1],
                            conditioning_list=[torch.from_numpy(cond[None][..., :streamwise_length, :]).to(device)], conditioning_scale_list=[1.], device=device, 
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask_[..., :streamwise_length, :]).to(device), patch_size=(56,56), ignore_index=1 )
            gen_sample.append(sample)
        else:
            prev_sample = gen_sample[-1][..., window_length:, :]
            sensor_slice = slice(streamwise_length + (i-1)*window_length, streamwise_length + (i)*window_length)
            sample = flow_padis_generalized(fm = FlowMatcher(1e-3), cfm_model=model,
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(1,streamwise_length,512), num_of_steps=200, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1, meas_func_2],
                            conditioning_list=[torch.from_numpy(cond[None][..., sensor_slice, :]).to(device), torch.from_numpy(prev_sample).to(device)],
                            conditioning_scale_list=[1.0, 1.0], device=device,
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask_[..., sensor_slice, :]).to(device), slice=slic, start=streamwise_length-window_length
                            ,patch_size=(56,56), ignore_index=1 )
            gen_sample.append(sample)

    samples.append(np.concat([gen if i==0 else  gen[..., streamwise_length-window_length:, :] for i, gen in enumerate(gen_sample) ], axis=2))
samples = np.concat(samples, axis=0)


# %%
samples_ = samples*std + m

# %%
np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp_case_V0/exp_{exp}/samples_cond_autoreg_{iteration}_sensors_{mask_.sum()}.npy", samples*std + m)


