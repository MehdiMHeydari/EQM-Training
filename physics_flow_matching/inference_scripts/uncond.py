import numpy as np
import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm
from torchdiffeq import odeint
from functools import partial
from torch.distributions import Chi2
from physics_flow_matching.utils.pre_procs_data import get_grad_energy, langevin_step, mala_condition
from physics_flow_matching.inference_scripts.utils import calculate_pad_size, extract_non_overlapping_patches, recombine_non_overlapping_patches

def infer(dims_of_img, total_samples, samples_per_batch,
          use_odeint, cfm_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, use_heavy_noise=False, 
          y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model_, t=torch.linspace(t_start, t_end, t_steps, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
    else:
        ode = NeuralODE(cfm_model_, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = ode_solver(y0)
            
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)       
#    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_rf_noise(dims_of_img, total_samples, samples_per_batch, use_odeint,
          model, device, t_start=5e-3, t_end=1,
          scale=True, m=None, std=None, t_steps=2, **kwargs):
    
    samples_list = []
    
    def cfm_model(t, x):
        assert t != 0., "Time cannot be zero"
        return (x - model(t, x))/t
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model, t=torch.linspace(t_start, t_end, 2, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
        
    else:
        ode = NeuralODE(cfm_model, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
             
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            traj = ode_solver(torch.randn(samples_size, device=device))
            
        out = traj[-1].detach().cpu().numpy()
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)
        
    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_em(dims_of_img, total_samples, samples_per_batch,
          use_odeint, em_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, use_heavy_noise=False, 
          y = None, y0_provided = False, y0= None, all_traj=False, 
          use_langevin=False, mala_correction=False, t_switch=0.8, eps_max=0.15, dt=0.01, M=300, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    
    def em_model_(t, x):
        return - get_grad_energy(x, em_model)
        
    samples_list = []
    if not use_langevin:
        if use_odeint:
            ode_solver_ = partial(odeint, func=em_model_, t=torch.linspace(t_start, t_end, t_steps, device=device), 
                                atol=1e-5, rtol=1e-5, 
                                method=kwargs["method"] if "method" in kwargs.keys() else None,
                                options=kwargs["options"] if "options" in kwargs.keys() else None)
            ode_solver = lambda x : ode_solver_(y0=x)
        else:
            ode = NeuralODE(em_model_, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
            ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
            ode_solver = lambda x: ode_solver_(x=x)
    else:
        ode_solver_ = partial(langevin_step, model=em_model, t_switch=t_switch, eps_max=eps_max, dt=dt)
        def ode_solver(x):
            traj = [x]
            for i in range(0, M): # langevin MCMC steps
                t = i*dt*torch.ones(x.shape[0]).to(device).unsqueeze(-1)
                x_prop = ode_solver_(x, t=t)
                alpha = torch.ones_like(t) if not mala_correction else torch.min(torch.ones_like(t), mala_condition(x, x_prop, em_model, t, t_switch, eps_max, dt))
                x = torch.where(torch.rand_like(t) <= alpha, x_prop, x) 
                traj.append(x)
            return torch.stack(traj, dim=0)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = ode_solver(y0)
            
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if samples_list[0].ndim == 2:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)       
#    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_patched_v1(dims_of_img, location ,total_samples, samples_per_batch,
                cfm_model, t_start, t_end,
                scale, device, m=None, std=None, t_steps=200, use_heavy_noise=False, 
                y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []
    
    time = torch.linspace(t_start, t_end, t_steps).to(device)
    dt = time[1] - time[0]
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 

        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()

            y0 = torch.concat([y0, location], dim=1)
            traj = []
            x = y0.clone()
            for t in time:
                v = cfm_model_(t, x)
                x[:, :3] += v[:, :3]*dt
                traj.append(x[:, :3])
                
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)    
    

def infer_patched(dims_of_img, patch_size, total_samples, samples_per_batch,
                cfm_model, t_start, t_end,
                scale, device, m=None, std=None, t_steps=200, use_heavy_noise=False, 
                y = None, y0_provided = False, y0= None, all_traj=False, ignore_index=3, **kwargs):

    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []
    
    time = torch.linspace(t_start, t_end, t_steps).to(device)
    dt = time[1] - time[0]
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
    
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = []
            x = y0.clone()
            
            B, _, H, W, _, _, _, _, pad_h, pad_w = calculate_pad_size(x, patch_size) 
            
            x_coord = (torch.linspace(-1, 1, H+2*pad_h)[None, None, :, None]).to(device) * torch.ones(B, 1, H+2*pad_h, W+2*pad_w).to(device)
        
            z_coord = (torch.linspace(-1, 1, W+2*pad_w)[None, None, None, :]).to(device) * torch.ones(B, 1, H+2*pad_h, W+2*pad_w).to(device)
            
            for t in time:
                offset_x, offset_y = (np.random.randint(low=0, high=pad_h+1), np.random.randint(low=0, high=pad_w+1))
                x_patchified = extract_non_overlapping_patches(x, (offset_x, offset_y), patch_size, x_coord, z_coord)
                v = cfm_model_(t, x_patchified)
                x_patchified[:, :ignore_index] += v[:, :ignore_index]*dt
                x = recombine_non_overlapping_patches(x_patchified[:, :ignore_index], dims_of_img, (pad_h, pad_w), (offset_x, offset_y), patch_size)
                traj.append(x)
                
        out = np.stack([tra.detach().cpu().numpy() for tra in traj]) if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)       
#    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_alt_joint(dims_of_img, total_samples, samples_per_batch,
                cfm_model, t_start, t_end,
                scale, device, m=None, std=None, t_steps=200, use_heavy_noise=False, 
                y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):

    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []
    
    time = torch.linspace(t_start, t_end, t_steps).to(device)
    dt = time[1] - time[0]
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
    
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = []
            x = y0.clone()
            for t in time[:-1]:
                v = cfm_model_(t, x)
                x[:, 3:] = x[:, 3:] +  v[:, 3:]*dt
                traj.append(x.detach().cpu().numpy())


        out =  np.stack([tra for tra in traj]) if all_traj else traj[-1]
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)
        

def infer_avg_vel(dims_of_img, total_samples, samples_per_batch,
                cfm_model, t_start, t_end,
                scale, device, m=None, std=None, t_steps=3, use_heavy_noise=False, 
                y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):

    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda r, t, x : cfm_model(r, t, x, y=y)
        
    samples_list = []
    
    time = torch.linspace(t_start, t_end, t_steps).to(device)
    dt = time[1] - time[0]
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
    
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = [y0.detach().cpu().numpy()]
            x = y0.clone()
            for t in range(t_steps - 1):
                x = x + cfm_model_(time[t+1], time[t], x) * (time[t+1] - time[t])
                traj.append(x.detach().cpu().numpy())


        out =  np.stack([tra for tra in traj]) if all_traj else traj[-1]
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)