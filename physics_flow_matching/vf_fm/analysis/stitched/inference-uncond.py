
import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.uncond import infer_stitched

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
start = np.load("/home/meet/storage/generative_stiching/channel_180_u_y1_221-259.npy")
end = np.load("/home/meet/storage/generative_stiching/channel_180_u_y199_221-259.npy")

# %%
std = np.load("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/stitched/exp_2/std.npy")

# %%
std_start, std_end = start.std(), end.std()

# %%
start_= start/std_start
end_ = end/std_end

# %% [markdown]
# ## Stitched Generation

# %%
exp = "stitched/exp_2"
iteration = 4
ot_cfm_model = UNetModel(dim=[3, 320, 200],
                        channel_mult=(1,2,4,4),
                        out_dim=1,
                        num_channels=128,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="40",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        class_cond=False,
                        num_classes=None,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/{exp}/saved_state/checkpoint_{iteration}.pth")
ot_cfm_model.load_state_dict(state["model_state_dict"])
ot_cfm_model.to(device)
ot_cfm_model.eval();

# %%
pytorch_total_params = sum(p.numel() for p in ot_cfm_model.parameters())

samples = []
for ind in range(0,3800,7):
    print(f"Loading start and end points from training data at index : {ind}")
    s = torch.from_numpy(start_[ind]).to(device)
    e = torch.from_numpy(end_[ind]).to(device)
    total_samples = 1
    samples_per_batch = 1
    t_steps=100

    samples.append(infer_stitched(len_of_traj=6, start=s, end=e, dims_of_img=(1, 320, 200), total_samples=total_samples, samples_per_batch=samples_per_batch,
                    use_odeint=True, cfm_model=ot_cfm_model, 
                    t_start=0., t_end=1., scale=False, device=device,
                    method="euler", use_heavy_noise=False, nu=3, y=None, options={"step_size":1/t_steps}, t_steps=t_steps))
    
samples = np.stack([sample[:, 0] for sample in samples])


np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/{exp}/samples_500_{iteration}_test.npy", samples*std)

