import torch as th
import torch.nn as nn

ACTS = {
    'relu': th.relu,
    'silu': nn.functional.silu,
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, act1=th.relu, act2=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Activation functions
        self.act1 = act1
        self.act2 = act2
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.act2(layer(x)) if i != 0 and self.act2 is not None else self.act1(layer(x))
        x = self.layers[-1](x)
        return x
    
class MLP_Wrapper(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, act1=th.relu, act2=None):
        super(MLP_Wrapper, self).__init__(input_dim, hidden_dims, output_dim, act1, act2)

    def forward(self, t, x, y=None):
        if t.dim() != x.dim():
            t = t.expand_as(x[..., 0:1])
        if y is not None:
            return super().forward(th.concat([t, x, y], dim=-1))
        else:
            return super().forward(th.concat([t, x], dim=-1))
        
class GuidedMLPWrapper(MLP_Wrapper):
    def __init__(self, input_dim, hidden_dims, output_dim, guide_func, act1=th.relu, act2=None, guide_scheduler_func=lambda t : 1.0, guide_scale=1.):
        super().__init__(input_dim, hidden_dims, output_dim, act1, act2)
        self.guide_func = guide_func
        self.guide_scale = guide_scale
        self.guide_scheduler_func = guide_scheduler_func
        
    def forward(self, t, x, y=None, *args, **kwargs):
        dx_dt = super().forward(t, x, y)
        if self.guide_func is not None:
            dx_dt = dx_dt + self.guide_func(t=t, x=x, v=dx_dt, cfm_model=super().forward) * self.guide_scale * self.guide_scheduler_func(t)
        return dx_dt
    
class EM_MLP_Wrapper(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, act1=th.relu, act2=None):
        super(EM_MLP_Wrapper, self).__init__(input_dim, hidden_dims, output_dim, act1, act2)

    def forward(self, x, y=None):
        if y is not None:
            return super().forward(th.concat([x, y], dim=-1))
        else:
            return super().forward(x)
        
class Avg_MLP_Wrapper(MLP):
    def __init__(self, input_dim, hidden_dims, output_dim, act1=th.relu, act2=None):
        super(Avg_MLP_Wrapper, self).__init__(input_dim, hidden_dims, output_dim, act1, act2)

    def forward(self, r, t, x, y=None):
        if t.dim() != x.dim():
            t = t.expand_as(x[..., 0:1])
        if r.dim() != x.dim():
            r = r.expand_as(x[..., 0:1])
        if y is not None:
            return super().forward(th.concat([r, t, x, y], dim=-1))
        else:
            return super().forward(th.concat([r, t, x], dim=-1))