import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import torch
import numpy as np
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.cond import infer_grad_generalized
from physics_flow_matching.inference_scripts.utils import grad_cost_func_generalized, sample_noise
from einops import rearrange

# %%
def read_data(file_path, nx, nz, nstep):
    data  = np.fromfile(file_path, dtype=np.float64, count=nx*nz*nstep).reshape((nx, nz, nstep), order='F')
    return data

nx_long, nz_long, nstep_long = 4800, 800, 500
Lx_long, Lz_long = 16*np.pi, 4*np.pi / 3

data = read_data("/storage/yi/Channel_M04/channel_re1000_16pi_y0001_180K-380K_dt=0.56_wallp.dat", nx_long, nz_long, nstep_long)

# %%
m = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/m_1000.npy")
std = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/std_1000.npy")
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
streamwise_length = 1200

mask = np.zeros((1,1,4800, 800))
mask[..., ::10, ::40] = 1.

# %%
exp = "baseline_1000"
iteration = 2
model = UNetModel(dim=[1, 256, 256],
                        channel_mult=(1, 1, 2, 4),
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
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/ablation_between_patched_and_full/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
model.load_state_dict(state["model_state_dict"])
model.to(device)
model.eval();

# %%
conds = rearrange(data_[..., ::5], "h w b -> b h w")[:, None] * mask

# %%
total_samples = 1
window_length = 1199
slic = slice(0, streamwise_length-window_length)

# %%
samples = []
for j, cond in enumerate(conds):
    print(f"Generating  sample : {j}")
    gen_sample = []
    for i in range(4800//window_length):
        if i == 0:
            sample = infer_grad_generalized(fm = FlowMatcher(1e-3), cfm_model=model,
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(1,streamwise_length,800), num_of_steps=100, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1],
                            conditioning_list=[torch.from_numpy(cond[None][..., :streamwise_length, :]).to(device)], conditioning_scale_list=[1.], device=device, 
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask[..., :streamwise_length, :]).to(device), swag=False)
            gen_sample.append(sample)
        else:
            prev_sample = gen_sample[-1][..., window_length:, :]
            sensor_slice = slice(streamwise_length + (i-1)*window_length, streamwise_length + (i)*window_length)
            sample = infer_grad_generalized(fm = FlowMatcher(1e-3), cfm_model=model,
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(1,streamwise_length,800), num_of_steps=100, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1, meas_func_2],
                            conditioning_list=[torch.from_numpy(cond[None][..., sensor_slice, :]).to(device), torch.from_numpy(prev_sample).to(device)],
                            conditioning_scale_list=[1.0, 1.0], device=device,
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask[..., sensor_slice, :]).to(device), slice=slic, start=streamwise_length-window_length, swag=False)
            gen_sample.append(sample)
    samples.append(np.concat([gen if i==0 else  gen[..., streamwise_length-window_length:, :] for i, gen in enumerate(gen_sample) ], axis=2))
    
samples = np.concat(samples, axis=0)

np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/patched_flow_matching/exps/wp/ablation_between_patched_and_full/exp_{exp}/samples_cond_autoreg_{iteration}_s{int(mask.sum())}.npy", (samples*std + m))