import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.mlp import EM_MLP_Wrapper as MLP
from physics_flow_matching.unet.mlp import ACTS
from torch.utils.data import DataLoader
from physics_flow_matching.multi_fidelity.synthetic.dataset import flow_guidance_dists
from physics_flow_matching.utils.train_em import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

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
    
    dataset = flow_guidance_dists(dist_name1=config.dataset.dist_name1,
                                  dist_name2=config.dataset.dist_name2, n=config.dataset.n, seed=config.dataset.seed)
    
    train_dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True)
        
    model = MLP(input_dim=config.mlp.input_dim,
                hidden_dims=config.mlp.hidden_dims,
                output_dim=config.mlp.output_dim,
                act1=ACTS[config.mlp.act1] if hasattr(config.mlp, 'act1') else ACTS['relu'],
                act2=ACTS[config.mlp.act2] if hasattr(config.mlp, 'act2') else None
                )

    model.to(dev)
    
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=config.FM.sigma)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None
    
    loss_fn = DD_loss
    
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
                return_noise=config.FM.return_noise,
                restart_epoch=config.restart_epoch,
                class_cond=config.mlp.class_cond if hasattr(config.mlp, 'class_cond') else False,
                contrastive_obj=config.contrastive.contrastive_obj if hasattr(config.contrastive, 'contrastive_obj') else False,
                mala_correction=config.contrastive.mala_correction if hasattr(config.contrastive, 'mala_correction') else False,
                M_lang=config.contrastive.M_lang if hasattr(config.contrastive, 'M_lang') else None,
                eps_max=config.contrastive.eps_max if hasattr(config.contrastive, 'eps_max') else None,
                t_switch=config.contrastive.t_switch if hasattr(config.contrastive, 't_switch') else None,
                dt=config.contrastive.dt if hasattr(config.contrastive, 'dt') else None,
                weight_cd=config.contrastive.weight_cd if hasattr(config.contrastive, 'weight_cd') else 0
                )

if __name__ == '__main__':
    main(sys.argv[1])