import torch as th

def DD_loss(pred: th.Tensor, target: th.Tensor):
    return ((pred - target)**2).mean()

def Contrastive_loss(pred: th.Tensor, target: th.Tensor, cont_target : th.Tensor, lmbda: float):
    return  DD_loss(pred, target) - lmbda * DD_loss(pred, cont_target)

def avg_vel_loss(pred: th.Tensor, target: th.Tensor, gamma: float=0.):
    delta = pred - target
    delta_l2_sq = delta.view(delta.shape[0], -1).pow(2).sum(dim=1)
    w = (1./ (delta_l2_sq + 1e-3)**(1 - gamma)).detach()
    loss = (w * delta_l2_sq).sum()
    return loss