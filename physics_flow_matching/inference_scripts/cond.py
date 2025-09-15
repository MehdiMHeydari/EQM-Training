import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from torchcfm.conditional_flow_matching import FlowMatcher, pad_t_like_x, ExactOptimalTransportConditionalFlowMatcher
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torch.optim import LBFGS, SGD
from physics_flow_matching.inference_scripts.utils import ssag_collect, extract_non_overlapping_patches, recombine_non_overlapping_patches, calculate_pad_size 
from typing import Union

def flowgrad(v_theta, cost_func, meas_func, measurement,
             device, num_of_samples, size,
             N=100, M=10, xi=5e-3, alpha=10.0, **kwargs):
    """
    Implements Algorithm 1 from the reference text. (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.pdf)

    Args:
      v_theta: The pre-trained ODE velocity function.
      N: The number of Euler discretization steps.
      M: The number of optimization iterations.
      lambd: Penalty coefficient.
      xi: Threshold for non-uniform discretization.
      alpha: Step size for gradient descent.

    Returns:
        The optimized output image x1.
    """
    
    L = lambda x : cost_func(meas_func, x, measurement, **kwargs)
    
    x0 = torch.randn(num_of_samples, *size, device=device)
    u = torch.zeros(N -1, *x0.shape, device=device)         # Initialize all the variables {u(k/N)}_{k=0}^{N-1} to zero.
    ts = torch.linspace(0, 1, N, device=device)
    dt = ts[1] - ts[0]
    
    pbar = tqdm(list(range(M)))
    
    for _ in pbar:
        # Simulate the PF ODE trajectory
        x = [x0]
        v = []
        for k, t in enumerate(ts[:-1]):
            with torch.no_grad():  # Pause gradient calculation during simulation
                xt = x[-1]
                x_k_plus_1 = xt + dt * (v_theta(t, xt) + u[k])
                x.append(x_k_plus_1)
                v.append(v_theta(t, xt))

        # Approximate with Non-uniform Discretization
        S = []
        for k in range(N - 1):
            if k == 0 or k == N - 2:
                S.append(torch.norm(v[k - 1] - v[k])**2 / torch.norm(v[k])**2)
            else:
                S.append(max(
                    torch.norm(v[k - 1] - v[k])**2 / torch.norm(v[k])**2,
                    torch.norm(v[k] - v[k + 1])**2 / torch.norm(v[k + 1])**2
                ))

        # Construct G
        G = [0]
        tj = 0
        while tj < 1:
            m = 1
            while tj + m*dt < 1 and sum(S[int(tj/dt):int(tj/dt) + m]) < xi:
                m += 1
            tj = tj + m*dt
            G.append(tj.item())

        # Fast Back-propagation
        x1 = x[-1]
        x1 = x1.requires_grad_()

        loss = L(x1)
        pbar.set_postfix({'distance': loss.item()}, refresh=False)
        grad_x1 = torch.autograd.grad(loss, x1)[0]
        grad_u = []

        for j in range(len(G) - 2, -1, -1):
            def phi_j_prime(x):
                t = ts[int(G[j]/dt)]
                return x + (G[j + 1] - G[j]) * (v_theta(t, x) + u[int(G[j]/dt)])

            _, grad_x1 = torch.autograd.functional.vjp(phi_j_prime, x[int(G[j]/dt)], grad_x1)
            grad_u.append((G[j + 1] - G[j]) * grad_x1)

        grad_u.reverse()

        # Update the variables
        for j in range(len(G) - 1):
            for k in range(int(G[j]/dt), int(G[j + 1]/dt)):
                u[k] = u[k] - alpha * grad_u[j]

    # # Generate the final output image
    # with torch.no_grad():
    #     xt = x0
    #     for t in ts[:-1]:
    #         vk = v_theta(t, xt)
    #         x_next = xt + dt * (vk + u[k])
    #         xt = x_next
            
    return xt.detach().cpu().numpy()#xt.detach().cpu().numpy()

def d_flow(
    cost_func,
    measurement_func,
    measurement,
    flow_model,
    initial_point,
    ode_solver=odeint,
    ode_solver_kwargs={"method": "midpoint", "t" : torch.linspace(0,1,2), "options": {"step_size": 1/6}},
    optimizer=LBFGS,
    optimizer_kwargs={"line_search_fn": "strong_wolfe"},
    max_iterations=10,
    full_output=False,
    regularize=False,
    reg_scale=1e-2,
    pretrain=False,
    **kwargs
):
    """
    Implements the D-Flow algorithm for controlled generation (https://arxiv.org/pdf/2402.14017).

    Args:
        cost_function: The cost function to be minimized.
        flow_model: The pre-trained flow model.
        initial_point: The initial point for optimization.
        ode_solver: The ODE solver to use. Default is torchdiffeq.odeint.
        ode_solver_kwargs: Keyword arguments for the ODE solver.
                            Default is {"method": "midpoint", "options": {"step_size": 1/6}}.
        optimizer: The optimization algorithm to use. Default is torch.optim.LBFGS.
        optimizer_kwargs: Keyword arguments for the optimizer.
                          Default is {"line_search_fn": "strong_wolfe"}.
        max_iterations: The maximum number of optimization iterations.

    Returns:
        The generated sample after optimization.
    """
    
    # dims = initial_point.dim() -1
    d = initial_point.view(initial_point.shape[0], -1).size(1)
    initial_point = torch.nn.Parameter(initial_point)
    
    def reg1(x):
        
        x = x.view(x.shape[0], -1)
        x_norm_sq = torch.clamp(0.5*torch.sum(x**2, dim=1),min=1e-6, max=1e6)
        x_norm = torch.sqrt(torch.sum(x**2, dim=1))#torch.sqrt(x_norm_sq)
        
        return (d - 1)*torch.log(x_norm + 1e-5) - x_norm_sq
    
    def reg2(x):
        
        x = x.view(x.shape[0], -1)
        x_norm_sq = torch.clamp(torch.mean(x**2, dim=1),min=1e-6, max=1e6)  #torch.clamp(torch.sum(x**2, dim=1),min=1e-6, max=1e6)  
        return x_norm_sq

    def closure():
        optimizer.zero_grad()
        generated_sample = ode_solver(
            flow_model, initial_point, **ode_solver_kwargs
        )[-1]
        loss = cost_func(measurement_func, generated_sample, measurement, **kwargs)
        if regularize:
            loss = loss + reg_scale*reg2(initial_point)
        loss.sum().backward()
        return loss.sum()
    
    # def avg_vel_closure(): ## optimization using source velocity
    #     time = ode_solver_kwargs["t"]
    #     time = time.to(initial_point)
    #     t_steps = len(time)
    #     optimizer.zero_grad()
    #     for t in range(t_steps - 1):
    #         if t == 0:
    #             x = initial_point + flow_model(time[t+1], time[t], initial_point) * (time[t+1] - time[t])
    #         else:
    #             x = x + flow_model(time[t+1], time[t], x) * (time[t+1] - time[t])
    #     loss = cost_func(measurement_func, x, measurement, **kwargs)
    #     loss.sum().backward()
    #     return loss.sum()
    
    optimizer = optimizer([initial_point], **optimizer_kwargs)
    
    pbar = tqdm(list(range(max_iterations)))

    # loss = torch.tensor(1e3)
    # loss_prev = torch.tensor(1e6)
    for _ in pbar: #while (loss).item()/initial_point.shape[0] > 1e-3  and (loss - loss_prev).abs().item()/loss_prev.item() > 1e-3:
    #    loss_prev = loss.clone()
       loss = optimizer.step(closure) #optimizer.step(avg_vel_closure)
       pbar.set_postfix({'distance': loss.item()}, refresh=False) #print("distance:", loss.item())#

    if not pretrain:
        with torch.no_grad():
            ## inference post optimization
        #     x = initial_point + flow_model(torch.tensor(.5).to(initial_point), torch.tensor(0.).to(initial_point), initial_point)*0.5
        #     return (x + flow_model(torch.tensor(1.).to(initial_point), torch.tensor(0.5).to(initial_point), x)*0.5).detach().cpu().numpy()
            return (ode_solver(flow_model, initial_point, **ode_solver_kwargs)[-1] if not full_output else ode_solver(flow_model, initial_point, **ode_solver_kwargs)).detach().cpu().numpy()
    else: 
        return initial_point.detach().cpu()

def d_flow_sgld(
    cost_func,
    measurement_func,
    measurement,
    flow_model,
    initial_point,
    ode_solver=odeint,
    ode_solver_kwargs={"method": "midpoint", "t" : torch.linspace(0,1,2), "options": {"step_size": 1/6}},
    max_iterations=100,
    step_size=0.01,
    noise_step = 1,
    start_collect_phase=10,
    regularize=False,
    reg_scale=1e-2,
    parallel=False,
    rms_prop_precond = False,
    beta = 0.99,
    delta = 1e-5,
    consider_loss_together=True,
    **kwargs
):
    """
    Implements a variant of the D-Flow algorithm with Stochastic Sample Averaging Gaussian

    Args:
        cost_function: The cost function to be minimized.
        flow_model: The pre-trained flow model.
        initial_point: The initial point for optimization.
        ode_solver: The ODE solver to use. Default is torchdiffeq.odeint.
        ode_solver_kwargs: Keyword arguments for the ODE solver.
                            Default is {"method": "midpoint", "options": {"step_size": 1/6}}.
        optimizer: The optimization algorithm to use. Default is torch.optim.LBFGS.
        optimizer_kwargs: Keyword arguments for the optimizer.
                          Default is {"line_search_fn": "strong_wolfe"}.
        max_iterations: The maximum number of optimization iterations.

    Returns:
        Mean, diagonal variance, and low rank approximation of covariance matrix on the base distribution space
    """
    
    # dims = initial_point.dim() -1
    pbar = tqdm(list(range(max_iterations)))
    collect = []#[initial_point.clone().cpu()]
    
    V = torch.zeros_like(initial_point)
    
    def reg2(x):
        if not parallel:
            x_norm_sq = torch.clamp(torch.sum(x**2), max=1e6)
            return x_norm_sq
        else:
            x = x.view(x.shape[0], -1)
            x_norm_sq = torch.clamp(torch.sum(x**2, dim=1),min=1e-6, max=1e6)
        
            return x_norm_sq
    
    for epoch in pbar:
        
        with torch.no_grad():
            
            with torch.enable_grad():
                initial_point.requires_grad_()
                generated_sample = ode_solver(flow_model, initial_point, **ode_solver_kwargs)[-1]
                loss = cost_func(measurement_func, generated_sample, measurement, **kwargs)
                
                if consider_loss_together:
                    if regularize:
                        loss = loss + reg_scale*reg2(initial_point)
                        
                    grad_init_point = torch.autograd.grad(loss if not parallel else loss.sum(), initial_point)[0]
                else:
                    grad_init_point = torch.autograd.grad(loss if not parallel else loss.sum(), initial_point)[0]
                    grad_init_point_prior = torch.autograd.grad(reg_scale*reg2(initial_point).sum(), initial_point)[0] if regularize else None
           
            if not rms_prop_precond:
                initial_point = initial_point - step_size[epoch] * grad_init_point + torch.sqrt(2 * step_size[epoch] * noise_step) * torch.randn_like(initial_point)
            else:
                V = beta * V + (1 - beta) * grad_init_point**2
                G = 1/(torch.sqrt(V) + delta[epoch])
                if not consider_loss_together and grad_init_point_prior is not None:
                    grad_init_point = grad_init_point + grad_init_point_prior
                initial_point = initial_point - step_size[epoch] * G * grad_init_point + torch.sqrt(2 * step_size[epoch] * G * noise_step) * torch.randn_like(initial_point)
                
            pbar.set_postfix({'distance': loss.item() if not parallel else loss.sum().item()}, refresh=False)
            if epoch >= start_collect_phase:
                collect.append(initial_point.detach().cpu())               
            
    return collect

def d_flow_ssag(
    cost_func,
    measurement_func,
    measurement,
    flow_model,
    initial_point,
    ode_solver=odeint,
    ode_solver_kwargs={"method": "midpoint", "t" : torch.linspace(0,1,2), "options": {"step_size": 1/6}},
    optimizer=SGD,
    optimizer_kwargs={"lr": 1e-3, "momentum": 0.9},
    max_iterations=100,
    start_collect_phase=10,
    cov_rank=20,
    momt_coll_freq=4,
    collect_phase_optimizer_kwargs = {"lr": 0.1, "momentum": 0.9},
    regularize=False,
    reg_scale=1e-2,
    parallel=False,
    **kwargs
):
    """
    Implements a variant of the D-Flow algorithm with Stochastic Sample Averaging Gaussian

    Args:
        cost_function: The cost function to be minimized.
        flow_model: The pre-trained flow model.
        initial_point: The initial point for optimization.
        ode_solver: The ODE solver to use. Default is torchdiffeq.odeint.
        ode_solver_kwargs: Keyword arguments for the ODE solver.
                            Default is {"method": "midpoint", "options": {"step_size": 1/6}}.
        optimizer: The optimization algorithm to use. Default is torch.optim.LBFGS.
        optimizer_kwargs: Keyword arguments for the optimizer.
                          Default is {"line_search_fn": "strong_wolfe"}.
        max_iterations: The maximum number of optimization iterations.

    Returns:
        Mean, diagonal variance, and low rank approximation of covariance matrix on the base distribution space
    """
    
    # dims = initial_point.dim() -1
    
    initial_point = torch.nn.Parameter(initial_point)
    optimizer = optimizer([initial_point], **optimizer_kwargs)
    pbar = tqdm(list(range(max_iterations)))
    
    first_momt = torch.zeros(initial_point.numel(), device=initial_point.device) if not parallel else torch.zeros_like(initial_point.view(initial_point.shape[0], -1), device=initial_point.device).requires_grad_(False)
    second_momt = torch.zeros(initial_point.numel(), device=initial_point.device) if not parallel else first_momt.clone().requires_grad_(False)
    num_collected = torch.tensor(0, device=initial_point.device)
    dev_matrix = torch.zeros(cov_rank, initial_point.numel(), device=initial_point.device) if not parallel else torch.zeros(cov_rank, *first_momt.shape, device=initial_point.device).requires_grad_(False)
    
    def reg2(x):
        if not parallel:
            x_norm_sq = torch.clamp(torch.sum(x**2), max=1e6)
            return x_norm_sq
        else:
            x = x.view(x.shape[0], -1)
            x_norm_sq = torch.clamp(torch.sum(x**2, dim=1),min=1e-6, max=1e6)
            # for _ in range(dims):
            #     x_norm_sq = x_norm_sq[..., None]
        
            return x_norm_sq
    
    for epoch in pbar:
        optimizer.zero_grad()
        generated_sample = ode_solver(flow_model, initial_point, **ode_solver_kwargs)[-1]
        loss = cost_func(measurement_func, generated_sample, measurement, **kwargs)
        if regularize:
            loss = loss + reg_scale*reg2(initial_point)
        loss.backward() if not parallel else loss.sum().backward()
        optimizer.step()
        pbar.set_postfix({'distance': loss.item() if not parallel else loss.sum().item()}, refresh=False)
        
        if epoch >= start_collect_phase:
            if epoch == start_collect_phase:
                for param_group in optimizer.param_groups:
                    param_group.update(**collect_phase_optimizer_kwargs)
                     
            first_momt, second_momt, num_collected, dev_matrix = ssag_collect(initial_point, first_momt, second_momt, epoch,
                                                                              momt_coll_freq, num_collected, dev_matrix, parallel)
            
    return first_momt, second_momt, dev_matrix, num_collected

def infer_grad(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, refine, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""
    
    # torch.manual_seed(seed=42)
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point#torch.randn(samples_size, device=device)
        conditioning_per_batch = conditioning #conditioning[i*samples_per_batch:(i+1)*samples_per_batch] #
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
                           
            for _ in range(refine): ##Picard Iteration
                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
                                                is_grad_free=False, grad={"t" : t, "v" : v},
                                                **kwargs)
                scaled_grad *= torch.linalg.norm(v.flatten())
                pbar.set_postfix({'distance': loss}, refresh=False)

                v = v - conditioning_scale*scaled_grad #beta
                x = x_fixed + v*dt #x + (v)*dt 
                
                x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)
    
def infer_gradfree(fm : FlowMatcher, cfm_model : torch.nn.Module,
                   samples_per_batch, total_samples, dims_of_img, 
                   num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                   sample_noise, use_heavy_noise, rf_start,
                   device, refine, start_provided=False, start_point=None, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : gradfree based algorithm"""
  
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point #torch.randn(samples_size, device=device)
        conditioning_per_batch = conditioning #conditioning[i*samples_per_batch:(i+1)*samples_per_batch]
        x_gauss = x.clone().detach()
        first_step = True
        
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

            x_fixed = x.clone().detach()
            
            for _ in range(refine): ##Picard Iteration

                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                if first_step:
                    first_step=False
                    x = x + v*dt
                    x = x.detach()
                    break
                
                else:
                    scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
                                                    is_grad_free=True, use_fd=False, grad_free={"a_t" : a_t, "b_t" : b_t, "x_gauss" : x_gauss},
                                                    **kwargs)
                    pbar.set_postfix({'distance': loss}, refresh=False)
                    scaled_grad *=  torch.linalg.norm(v.flatten())
                    v = v - conditioning_scale*scaled_grad #beta
                    x = x_fixed + (v)*dt #
                
                x = x.detach()
                      
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

def infer_parallel(cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, sample_noise, use_heavy_noise, rf_start, class_index=None, all_traj=False,
                start_provided=False, is_grad_free=False, start_point=None, solver=None, **kwargs):
    
    # torch.manual_seed(seed=42)
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    # dt = ts[1] - ts[0]
        
    samples = []
    cfm_model.guide_func = partial(grad_cost_func, meas_func=meas_func, is_grad_free=is_grad_free, **kwargs)
    cfm_model.guide_scale = conditioning_scale
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point[i*samples_per_batch:(i+1)*samples_per_batch]
        assert x.shape[0] == samples_per_batch, f"Batch size mismatch: {x.shape[0]} != {samples_per_batch}"
        conditioning_per_batch = conditioning[i*samples_per_batch:(i+1)*samples_per_batch]
                
        cfm_model.guide_func = partial(cfm_model.guide_func, measurement=conditioning_per_batch)
        
        cfm_model.forward = partial(cfm_model.forward, y=class_index.repeat(samples_per_batch)) if hasattr(cfm_model, "num_classes") and cfm_model.num_classes is not None else cfm_model
        
        node = NeuralODE(cfm_model,
                     solver="euler" if solver is None else solver,
                     sensitivity="adjoint",
                     atol=1e-4,
                     rtol=1e-4)

        with torch.no_grad():
            traj = node.trajectory(x.to(device), t_span=ts)

        samples.append(traj[-1].cpu().numpy() if not all_traj else traj.cpu().numpy())
                
    return np.concatenate(samples)  if not all_traj else np.concatenate(samples, axis=1)

def infer_grad_generalized(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func_list, conditioning_list, conditioning_scale_list,
                device, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""
    
    # torch.manual_seed(seed=42)
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point
        conditioning_per_batch_list = [conditioning[i*samples_per_batch:(i+1)*samples_per_batch] for conditioning in conditioning_list]
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
                           
            x = x.requires_grad_()
            v = cfm_model(t, x)
            
            scaled_grad_list, loss_list = grad_cost_func(meas_func_list, x, conditioning_per_batch_list, 
                                            is_grad_free=False, grad={"t" : t, "v" : v},
                                            **kwargs)
            
            total_grad = 0.
            total_loss = 0.
            v_norm = torch.linalg.norm(v.flatten())
            for scaled_grad, loss, conditioning_scale in zip(scaled_grad_list, loss_list, conditioning_scale_list):
                scaled_grad *= v_norm
                total_grad += conditioning_scale*scaled_grad
                total_loss += loss
            pbar.set_postfix({'distance': total_loss}, refresh=False)
            v = v - total_grad
            x = x_fixed + v*dt #x + (v)*dt 
            
            x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

def flow_dps(fm : FlowMatcher, cfm_model : torch.nn.Module,
             samples_per_batch, total_samples, dims_of_img, 
             num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
             device, refine, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, **kwargs):
    
    """https://arxiv.org/pdf/2503.08136"""
    
    pass

def flow_daps(fm : Union[FlowMatcher, ExactOptimalTransportConditionalFlowMatcher], cfm_model : torch.nn.Module,
              samples_per_batch, total_samples, dims_of_img,
              num_of_steps, meas_func, conditioning, beta,
              eta, r, langevin_mc_steps,
              device, sample_noise, use_heavy_noise, 
              start_provided=False, start_point=None, **kwargs):

    # torch.manual_seed(seed=42)
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 

        x0_lang_up = None
        eps_hat = None       
        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point[i*samples_per_batch:(i+1)*samples_per_batch]
        conditioning_per_batch = conditioning[i*samples_per_batch:(i+1)*samples_per_batch]
        
        pbar = tqdm(ts[:-1])
        for i, t in enumerate(pbar):
            if i != 0:
                x = fm.sample_xt(eps_hat, x0_lang_up, t, torch.randn_like(x))
            with torch.no_grad():
                v = cfm_model(t, x)
                x0_hat = x + (1 - t)*v #could add a multi-step ODE solver, for more accuracy
                eps_hat = x - t*v
            x0_lang_up = x0_hat.detach().clone()
            
            for _ in range(langevin_mc_steps):
                x0_lang_up = x0_lang_up.requires_grad_()
                with torch.enable_grad():
                    gaussian_err = ((x0_lang_up - x0_hat)**2).sum()/(2*(r[i])**2)
                    gauss_grad = torch.autograd.grad(gaussian_err, x0_lang_up)[0]
                    
                    meas_err = ((meas_func(x0_lang_up, **kwargs) - conditioning_per_batch)**2).sum()/(2*beta**2)
                    meas_grad = torch.autograd.grad(meas_err, x0_lang_up)[0]
                
                x0_lang_up = x0_lang_up - eta[i] * gauss_grad - eta[i] * meas_grad + (2*eta[i])**0.5 * torch.randn_like(x0_lang_up)
                x0_lang_up = x0_lang_up.detach()
            pbar.set_postfix({'distance': meas_err.item()*(2*beta**2)}, refresh=False)

        samples.append(x0_lang_up.detach().cpu().numpy())
    
    return np.concat(samples)

def flow_padis(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, refine, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, ignore_index=3, **kwargs):
    """https://arxiv.org/pdf/2406.02462 : patched flow matching FMPS"""

    
    # torch.manual_seed(seed=42)
    patch_size = kwargs["patch_size"]
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point#torch.randn(samples_size, device=device)
        conditioning_per_batch = conditioning #conditioning[i*samples_per_batch:(i+1)*samples_per_batch] #
        
        B, C, H, W, patch_h, patch_w, num_patch_H, num_patch_W, pad_h, pad_w = calculate_pad_size(x, patch_size)
        
        x_coord = (torch.linspace(-1, 1, H+2*pad_h)[None, None, :, None]).to(device) * torch.ones(B,1,H+2*pad_h,
                                                                                                      W+2*pad_w).to(device)
        
        z_coord = (torch.linspace(-1, 1, W+2*pad_w)[None, None, None, :]).to(device) * torch.ones(B,1,H+2*pad_h,
                                                                                                      W+2*pad_w).to(device)        
        
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
                           
            for _ in range(refine): ##Picard Iteration
                offset = (np.random.randint(0, high=pad_h+1), np.random.randint(0, high=pad_w+1))
                x = x.requires_grad_()
                x_patches = extract_non_overlapping_patches(x, offset=offset, patch_size=patch_size, x_coord=x_coord, z_coord=z_coord)
                
                v_patches = cfm_model(t, x_patches)
                
                # x = recombine_non_overlapping_patches(x_patches, dims_of_img, (pad_h,pad_w), offset, patch_size)
                v_recons = recombine_non_overlapping_patches(v_patches, dims_of_img, (pad_h,pad_w), offset, patch_size)
                
                v = v_recons[:, :ignore_index]
                # x, v = x[:, :ignore_index], v_recons[:, :ignore_index]
                
                scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
                                                is_grad_free=False, grad={"t" : t, "v" : v},
                                                **kwargs)
                
                scaled_grad *= torch.linalg.norm(v.flatten())
                pbar.set_postfix({'distance': loss}, refresh=False)

                v = v - conditioning_scale*scaled_grad #beta
                x = x_fixed + v*dt #x + (v)*dt 
                
                x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

def flow_padis_generalized(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func_list, conditioning_list, conditioning_scale_list,
                device, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, ignore_index=3, **kwargs):

    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""
    
    # torch.manual_seed(seed=42)
    patch_size = kwargs["patch_size"]
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point
        
        B, C, H, W, patch_h, patch_w, num_patch_H, num_patch_W, pad_h, pad_w = calculate_pad_size(x, patch_size)
        
        x_coord = (torch.linspace(-1, 1, H+2*pad_h)[None, None, :, None]).to(device) * torch.ones(B,1,H+2*pad_h,
                                                                                                      W+2*pad_w).to(device)
        
        z_coord = (torch.linspace(-1, 1, W+2*pad_w)[None, None, None, :]).to(device) * torch.ones(B,1,H+2*pad_h,
                                                                                                      W+2*pad_w).to(device)        
        
        conditioning_per_batch_list = [conditioning[i*samples_per_batch:(i+1)*samples_per_batch] for conditioning in conditioning_list]
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            offset = (np.random.randint(0, high=pad_h+1), np.random.randint(0, high=pad_w+1))
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
            x = x.requires_grad_()
            
            x_patches = extract_non_overlapping_patches(x, offset=offset, patch_size=patch_size, x_coord=x_coord, z_coord=z_coord)

            v_patches = cfm_model(t, x_patches)
            
            # x = recombine_non_overlapping_patches(x, dims_of_img, (pad_h,pad_w), offset, patch_size)
            v_recons = recombine_non_overlapping_patches(v_patches, dims_of_img, (pad_h,pad_w), offset, patch_size)
            
            v = v_recons[:, :ignore_index]
            #x, v = x[:, :ignore_index], v[:, :ignore_index]
            
            scaled_grad_list, loss_list = grad_cost_func(meas_func_list, x, conditioning_per_batch_list, 
                                            is_grad_free=False, grad={"t" : t, "v" : v},
                                            **kwargs)
            
            total_grad = 0.
            total_loss = 0.
            v_norm = torch.linalg.norm(v.flatten())
            for scaled_grad, loss, conditioning_scale in zip(scaled_grad_list, loss_list, conditioning_scale_list):
                scaled_grad *= v_norm
                total_grad += conditioning_scale*scaled_grad
                total_loss += loss
            pbar.set_postfix({'distance': total_loss}, refresh=False)
            v = v - total_grad
            x = x_fixed + v*dt #x + (v)*dt 
            
            x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

# def infer_grad_fd(fm : FlowMatcher, cfm_model : torch.nn.Module,
#                    samples_per_batch, total_samples, dims_of_img, 
#                    num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
#                    sample_noise, use_heavy_noise, rf_start,
#                    device, refine, **kwargs):
#     """Use finite difference to approx the u(t,x)"""
  
#     start = 5e-3 if rf_start else 0 
#     ts = torch.linspace(start, 1, num_of_steps, device=device)
#     dt = ts[1] - ts[0]
    
#     samples_per_batch = 1
    
#     samples = []
    
#     for i in range(total_samples//samples_per_batch):
#         samples_size = samples_per_batch
#         if i == total_samples//samples_per_batch - 1:
#             samples_size = samples_per_batch + total_samples%samples_per_batch
#         samples_size = (samples_size,) + dims_of_img 
        
#         x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"])#torch.randn(samples_size, device=device)
#         conditioning_per_batch = conditioning[i*samples_per_batch:(i+1)*samples_per_batch]
#         first_step = True
        
#         pbar = tqdm(ts[:-1])        
#         for t in pbar:
#             _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
#             beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
#                              pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

#             x_fixed = x.clone().detach()
            
#             for _ in range(refine): ##Picard Iteration

#                 x = x.requires_grad_()
#                 v = cfm_model(t, x)
                
#                 if first_step:
#                     first_step=False
#                     x = x + v*dt
#                     x = x.detach()
#                     x_prev = x_fixed.clone().detach()
#                     break
                
#                 else:
#                     scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
#                                                     is_grad_free=True, use_fd=True, grad_free={"x_prev" : x_prev, "dt": dt, "t" : t},
#                                                     **kwargs)
#                     pbar.set_postfix({'distance': loss}, refresh=False)
#                     scaled_grad *=  torch.linalg.norm(v.flatten())
#                     v = v - conditioning_scale*scaled_grad #beta
#                     x = x_fixed + (v)*dt #
                
#                 x = x.detach()
#                 x_prev = x_fixed.clone().detach()
                     
#         samples.append(x.cpu().numpy())
        
#     return np.concatenate(samples)

# def infer_gradfree_ho(fm : FlowMatcher, cfm_model : torch.nn.Module,
#                    samples_per_batch, total_samples, dims_of_img, 
#                    num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
#                    sample_noise, use_heavy_noise, rf_start,
#                    device, **kwargs):
#     """gradfree based algorithm with higher order method like Heun's, RK4 method"""
    
#     def heun_step(t, x, dt):
#         k1 = cfm_model(t, x)
#         if t + dt >= 1.:
#             return k1
#         k2 = cfm_model(t + dt, x + k1*dt)
#         return 0.5*(k1 + k2)
    
#     def rk4_step(t, x, dt):
#         k1 = cfm_model(t, x)
#         if t + dt >= 1.:
#             return k1
#         k2 = cfm_model(t + 0.5*dt, x + 0.5*k1*dt)
#         k3 = cfm_model(t + 0.5*dt, x + 0.5*k2*dt)
#         k4 = cfm_model(t + dt, x + k3*dt)
#         return (k1 + 2*k2 + 2*k3 + k4)/6.
  
#     start = 5e-3 if rf_start else 0 
#     ts = torch.linspace(start, 1, num_of_steps, device=device)
#     dt = ts[1] - ts[0]
    
#     samples_per_batch = 1
    
#     samples = []
    
#     for i in range(total_samples//samples_per_batch):
#         samples_size = samples_per_batch
#         if i == total_samples//samples_per_batch - 1:
#             samples_size = samples_per_batch + total_samples%samples_per_batch
#         samples_size = (samples_size,) + dims_of_img 
        
#         x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"])#torch.randn(samples_size, device=device)
#         conditioning_per_batch = conditioning[i*samples_per_batch:(i+1)*samples_per_batch]
#         x_gauss = x.clone().detach()
#         first_step = True
        
#         pbar = tqdm(ts[:-1])        
#         for t in pbar:
#             _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
#             beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
#                              pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

            
#             x = x.requires_grad_()
#             v = rk4_step(t, x, dt)#heun_step(t, x, dt)
            
#             if first_step:
#                 first_step=False
#                 x = x + v*dt
            
#             else:
#                 scaled_grad, mse_loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
#                                                 is_grad_free=True, use_fd=False, grad_free={"a_t" : a_t, "b_t" : b_t, "x_gauss" : x_gauss},
#                                                 **kwargs)
#                 pbar.set_postfix({'distance': mse_loss}, refresh=False)
#                 scaled_grad *=  torch.linalg.norm(v.flatten())
#                 v = v - conditioning_scale*scaled_grad #beta
#                 x = x + (v)*dt #
            
#             x = x.detach()
                      
#         samples.append(x.cpu().numpy())
        
#     return np.concatenate(samples)

# def oc_flow(f_p, cost_fn, meas_fn, measurement,
#             max_iterations, lr,
#             beta, num_of_steps, size,
#             device, control_update_freq=1, **kwargs):
#     """
#     Implements Algorithm 1: OC-Flow on Euclidean Space. (https://arxiv.org/pdf/2410.18070)
#     """

#     reward_fn = lambda x, x_p: -(cost_fn(meas_fn, x, measurement, **kwargs) + ((x - x_p)**2).mean())
#     ts = torch.linspace(0, 1, num_of_steps, device=device)
#     dt = ts[1] - ts[0]

#     x0 = torch.randn(1, *size, device=device) 

#     # Initialize control terms
#     theta = torch.zeros(num_of_steps//control_update_freq, *x0.shape, device=device)  

#     # Optimization loop
#     for _ in tqdm(range(max_iterations)):
#         with torch.no_grad():
#             # Solve for the state trajectory (Euler discretization)
#             x_prev = x0.clone()
#             x_list = [x_prev]
#             j = 0  # Index for control terms
#             for i, t in enumerate(ts[:-1]):  # Iterate up to the second-to-last element
#                 if i % control_update_freq == 0:
#                     x_t = x_prev + (f_p(t, x_prev) + theta[j]) * dt
#                 else:
#                     x_t = x_prev + f_p(t, x_prev) * dt
#                 x_prev = x_t # Update x_prev for the next iteration
#                 x_list.append(x_prev)
#                 if (i + 1) % control_update_freq == 0:  # Update control index
#                     j += 1
            
#             if theta.sum() == 0:
#                 x_p = x_list[-1].clone().detach()
                
#         # Calculate the gradient (adjoint method)
#         x_prev = x_list.pop().requires_grad_()
#         grad_x_t = torch.autograd.grad(reward_fn(x_prev, x_p), x_prev, retain_graph=False)[0]
#         del x_prev
#         grad_x = [grad_x_t]
#         for t in (reversed(ts[:-1])):  # Iterate in reverse from the second element
#             x_t = x_list.pop()
#             grad_x_t = torch.autograd.functional.vjp(f_p, inputs=(t, x_t), v=grad_x_t)[1][1] ## Gradient blows up for the inpainting task
#             grad_x.append(grad_x_t)
#             del x_t
#         # Update control (no need to store the entire grad_x)
#         grad_x = torch.stack(grad_x[::-1])
#         theta = beta * theta + lr * grad_x[::control_update_freq]

#     # Solve for the final state trajectory with optimized control
#     with torch.no_grad():
#         x_prev = x0
#         j = 0  # Index for control terms
#         for i, t in enumerate(ts[:-1]):  # Iterate up to the second-to-last element
#             x_t = x_prev + (f_p(t, x_prev) + theta[j]) * dt
#             x_prev = x_t # Update x_prev for the next iteration
#             if (i + 1) % control_update_freq == 0:  # Update control index
#                 j += 1
#     return x_prev.detach().cpu().numpy()

# def d_flow_ssag_high_dim(
#     cost_func,
#     measurement_func,
#     measurement,
#     flow_model,
#     initial_point,
#     ode_solver=odeint,
#     ode_solver_kwargs={"method": "midpoint", "t" : torch.linspace(0,1,2), "options": {"step_size": 1/6}},
#     max_iterations=100,
#     optimizer_kwargs={"line_search_fn": "strong_wolfe"},
#     start_collect_phase=10,
#     cov_rank=20,
#     momt_coll_freq=4,
#     collect_phase_optimizer = SGD,
#     collect_phase_optimizer_kwargs = {"lr": 0.1, "momentum": 0.9},
#     regularize=False,
#     reg_scale=1e-2,
#     parallel=False,
#     **kwargs
# ):
#     """
#     Implements a variant of the D-Flow algorithm with Stochastic Sample Averaging Gaussian for Higher Dimensional Data
#     """
    
#     initial_point_close = d_flow(cost_func=cost_func, measurement_func=measurement_func, measurement=measurement, flow_model=flow_model,
#                                  initial_point=initial_point, 
#                                  optimizer_kwargs=optimizer_kwargs, max_iterations= start_collect_phase,
#                                  ode_solver=ode_solver, ode_solver_kwargs=ode_solver_kwargs,
#                                  full_output=False, regularize=regularize, reg_scale=reg_scale, pretrain=True, **kwargs)
        
#     return d_flow_ssag(cost_func=cost_func, measurement_func= measurement_func, measurement=measurement, flow_model=flow_model, 
#                        initial_point= initial_point_close,
#                        optimizer=collect_phase_optimizer,
#                        ode_solver=ode_solver, ode_solver_kwargs=ode_solver_kwargs,                                          
#                        optimizer_kwargs=collect_phase_optimizer_kwargs, collect_phase_optimizer_kwargs=collect_phase_optimizer_kwargs,
#                        max_iterations=max_iterations - start_collect_phase, start_collect_phase=0, cov_rank=cov_rank, momt_coll_freq=momt_coll_freq,
#                        regularize=regularize, reg_scale=reg_scale,parallel=parallel, **kwargs)


def infer_grad_dpmc(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, refine, sample_noise, use_heavy_noise, rf_start, eta,start_provided=False, start_point=None, 
                noise_start_frac= 0.1,
                noise_end_frac = 0.9,
                **kwargs):
    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""
    
    # torch.manual_seed(seed=42)
    start = 5e-3 if rf_start else 0 
    ts = torch.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    noise_start_step = int(num_of_steps*noise_start_frac)
    noise_end_step  = int(num_of_steps*noise_end_frac)
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point[i*samples_per_batch:(i+1)*samples_per_batch]#torch.randn(samples_size, device=device)
        conditioning_per_batch = conditioning[i*samples_per_batch:(i+1)*samples_per_batch] #conditioning[i*samples_per_batch:(i+1)*samples_per_batch] #
        pbar = tqdm(ts[:-1])        
        for step_idx, t in enumerate(pbar):
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
            if noise_start_step <= step_idx < noise_end_step:
                refine = refine
            else:
                refine = 1
            for _ in range(refine): ##Picard Iteration
                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                
                scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
                                                is_grad_free=False, grad={"t" : t, "v" : v},
                                                **kwargs)
                scaled_grad *= torch.linalg.norm(v.flatten())
                pbar.set_postfix({'distance': loss}, refresh=False)

                v = v - conditioning_scale*scaled_grad  #beta
                if noise_start_step <= step_idx < noise_end_step:
                    scale = (2 * eta[step_idx]).sqrt()
                    x = x + v*dt + scale*torch.randn(x.shape, device=x.device)#x + (v)*dt 
                else:
                    x = x + v*dt
                x = x.detach()
        
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)
