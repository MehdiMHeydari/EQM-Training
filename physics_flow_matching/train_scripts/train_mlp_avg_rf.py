import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.mlp import Avg_MLP_Wrapper as MLP
from torch.utils.data import DataLoader
from physics_flow_matching.multi_fidelity.synthetic.dataset import flow_guidance_dists
from physics_flow_matching.utils.train_avg_rf import train_model
from physics_flow_matching.utils.obj_funcs import avg_vel_loss
from torchcfm.conditional_flow_matching import RectifiedFlow
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from functools import partial

def create_dir(path, config):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        assert config.restart != False, "Are you restarting?"
        print(f"Directory '{path}' already exists.")

def main(config_path):

    config = OmegaConf.load(config_path)
    
    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)
    
    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)
    
    dev = th.device(config.device)
    
    writer = SummaryWriter(log_dir=logpath)
    
    dataset =  flow_guidance_dists(dist_name1=config.dataset.dist_name1,
                                   dist_name2=config.dataset.dist_name2, n=config.dataset.n,
                                   seed=config.dataset.seed, 
                                   normalize=config.dataset.normalize if hasattr(config.dataset, 'normalize') else False,
                                   contrastive=config.dataset.contrastive if hasattr(config.dataset, 'contrastive') else False,
                                   flip=config.dataset.flip if hasattr(config.dataset, 'flip') else False)
    
    train_dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True)
        
    model = MLP(input_dim=config.mlp.input_dim,
                hidden_dims=config.mlp.hidden_dims,
                output_dim=config.mlp.output_dim
                )

    model.to(dev)
    
    FM = RectifiedFlow(add_heavy_noise=config.FM.add_heavy_noise if hasattr(config.FM, 'add_heavy_noise') else False,
                       nu = config.FM.nu if hasattr(config.FM, 'nu') else 0)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
    loss_fn = partial(avg_vel_loss, gamma= config.gamma if hasattr(config, 'gamma') else 0.)
    
    train_model(model=model,
                FM=FM,
                train_dataloader=train_dataloader,
                optimizer=optim,
                sched=sched,
                loss_fn=loss_fn,
                writer=writer,
                num_epochs=config.num_epochs,
                print_epoch_int=config.print_epoch_int,
                save_epoch_int=config.save_epoch_int,
                print_within_epoch_int=config.print_with_epoch_int,
                path=savepath,
                device=dev,
                restart=config.restart,
                restart_epoch=config.restart_epoch)
    
if __name__ == '__main__':
    main(sys.argv[1])