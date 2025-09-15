import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.ebm_net import EBM_Wrapper as EBM
from torch.utils.data import DataLoader
# from physics_flow_matching.multi_fidelity.synthetic.dataset import flow_guidance_dists
from physics_flow_matching.multi_fidelity.synthetic.dataset import Syn_Data_FM
from physics_flow_matching.utils.train_ebm import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

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
    
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    dataset = datasets.MNIST(root='/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/data', train=True, download=True, transform=transform)
    
    train_dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True)
        
    model = EBM(
                cnn_add_max_pool_layer=config.net.cnn_add_max_pool_layer,
                cnn_channel_list=config.net.cnn_channel_list,
                mlp_feature_list=config.net.mlp_feature_list
               )
    
    model.to(dev)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
        
    train_model(model=model,
                train_dataloader=train_dataloader,
                optimizer=optim,
                sched=sched,
                writer=writer,
                num_epochs=config.num_epochs,
                print_epoch_int=config.print_epoch_int,
                save_epoch_int=config.save_epoch_int,
                print_within_epoch_int=config.print_with_epoch_int,
                path=savepath,
                device=dev,
                mala_correction=config.contrastive.mala_correction if hasattr(config.contrastive, 'mala_correction') else False,
                M_lang=config.contrastive.M_lang if hasattr(config.contrastive, 'M_lang') else 200,
                eps_max=config.contrastive.eps_max if hasattr(config.contrastive, 'eps_max') else 1e-2,
                t_switch=config.contrastive.t_switch if hasattr(config.contrastive, 't_switch') else 0.,
                weight_alpha= config.contrastive.weight_alpha if hasattr(config.contrastive, 'weight_alpha') else 1e-2,
                clip_grad= config.contrastive.clip_grad  if hasattr(config.contrastive, 'clip_grad') else False,
                clip_val= config.contrastive.clip_val  if hasattr(config.contrastive, 'clip_val') else 1e-2,
                dt=config.contrastive.dt if hasattr(config.contrastive, 'dt') else 1e-2,
                restart=config.restart,
                restart_epoch=config.restart_epoch,
                class_cond=config.net.class_cond if hasattr(config.net, 'class_cond') else False)

if __name__ == '__main__':
    main(sys.argv[1])