import torch as th
from torchcfm.conditional_flow_matching import FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from typing import Union

def get_batch(FM : Union[FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher], x0 : th.Tensor, x1 : th.Tensor, return_noise=False):
        
    if return_noise:
        t, xt, ut, noise = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
    if return_noise:
        return t[..., None], xt, ut, noise
    return t[..., None], xt, ut

def get_batch_cont(FM : Union[FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher], x0 : th.Tensor, x1 : th.Tensor, x0_cont : th.Tensor, x1_cont : th.Tensor, return_noise=False):
        
    if return_noise:
        t, xt, ut, noise = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        _, __, ut_cont, noise_cont  = FM.sample_location_and_conditional_flow(x0_cont, x1_cont, t=t, return_noise=return_noise)
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        _, __, ut_cont = FM.sample_location_and_conditional_flow(x0_cont, x1_cont, t=t, return_noise=return_noise)

    if return_noise:
        return t[..., None], xt, ut, ut_cont, (noise, noise_cont)
    return t[..., None], xt, ut, ut_cont

def get_batch_avg(FM : RectifiedFlow, x0 : th.Tensor, x1 : th.Tensor):

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=False)
    r = th.empty_like(t)
    # r[:int(t.shape[0]*0.25)] = th.rand(int(t.shape[0]*0.25)).to(t.device)
    # r[int(t.shape[0]*0.25):] = t[int(t.shape[0]*0.25):] * th.rand(t.shape[0] - int(t.shape[0]*0.25)).to(t.device)
    r  = th.rand(t.shape).to(t.device)
    r = th.where(r > t, r, t)

    return t[..., None], r[..., None], xt, ut

#######################################################################################################################

def get_grad_energy(x, model, retain_graph=False, create_graph=False):
    with th.enable_grad():
        x = x.requires_grad_(True)
        v = model(x)
        return th.autograd.grad(v.sum(), x,retain_graph=retain_graph, create_graph=create_graph)[0] #grad_outputs=th.ones_like(v).to(v)

def get_epsilon(t, t_switch, eps_max, data_dim_except_batch=1):
    eps = th.where(t < t_switch, th.zeros_like(t),
                    th.where(t < 1, 
                    eps_max * (t - t_switch) / (1 - t_switch), eps_max * th.ones_like(t)))
    for _ in range(data_dim_except_batch):
        eps = eps.unsqueeze(-1)
    return eps

def langevin_step(x, model, t, t_switch, eps_max, dt, data_dim_except_batch=1, retain_graph=False, create_graph=False):
    noise = th.randn_like(x)
    grad_v = get_grad_energy(x, model, retain_graph=retain_graph, create_graph=create_graph)
    eps = get_epsilon(t, t_switch, eps_max, data_dim_except_batch)
    
    x = x - dt * grad_v + th.sqrt(2 * dt * eps) * noise
    return x

def log_q_dist(x1, x2, grad_x2, dt):
    return (-0.25 * (1/dt) * ((x1 - x2 + dt*grad_x2)**2).flatten(start_dim=1).sum(dim=1)).unsqueeze(-1)

def mala_condition(x, x_proposal, model, t, t_switch, eps_max, dt):
    log_pi_x_proposal = - model(x_proposal) / eps_max
    log_pi_x = - model(x) / eps_max
    
    log_q_x = log_q_dist(x, x_proposal, get_grad_energy(x_proposal, model), dt)
    log_q_x_proposal = log_q_dist(x_proposal, x, get_grad_energy(x, model), dt)
    
    return th.exp(log_pi_x_proposal - log_pi_x + log_q_x - log_q_x_proposal).squeeze()