import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel
from physics_flow_matching.utils.dataloader import get_joint_loaders
from physics_flow_matching.utils.dataset import DATASETS
from physics_flow_matching.utils.train_eqm import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torchcfm.conditional_flow_matching import EquilibriumMatching
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def create_dir(path, config):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        if hasattr(config, 'restart') and config.restart:
            print(f"Directory '{path}' already exists. Restarting training...")
        else:
            print(f"Directory '{path}' already exists. Continuing...")

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

    # Check if using HDF5 dataset (DarcyFlow) or traditional NPY files
    dataset_name = config.dataloader.dataset
    if dataset_name == "DarcyFlow":
        # For DarcyFlow, instantiate dataset directly with HDF5 file
        dataset_kwargs = {
            'hdf5_path': config.dataloader.datapath,
            'contrastive': config.dataloader.contrastive,
        }
        # Add optional parameters if present in config
        if hasattr(config.dataloader, 'input_key'):
            dataset_kwargs['input_key'] = config.dataloader.input_key
        if hasattr(config.dataloader, 'output_key'):
            dataset_kwargs['output_key'] = config.dataloader.output_key
        if hasattr(config.dataloader, 'normalize'):
            dataset_kwargs['normalize'] = config.dataloader.normalize
        if hasattr(config.dataloader, 'use_eqm_format'):
            dataset_kwargs['use_eqm_format'] = config.dataloader.use_eqm_format

        dataset = DATASETS[dataset_name](**dataset_kwargs)
        train_dataloader = DataLoader(dataset,
                                     batch_size=config.dataloader.batch_size,
                                     shuffle=True)
    else:
        # Use traditional NPY file loader
        train_dataloader = get_joint_loaders(vf_paths=config.dataloader.datapath,
                                            batch_size=config.dataloader.batch_size,
                                            dataset_=DATASETS[dataset_name],
                                            contrastive=config.dataloader.contrastive)

    model = UNetModel(dim=config.unet.dim,
                      out_channels=config.unet.out_channels,
                      channel_mult=config.unet.channel_mult,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film,
                      class_cond= config.unet.class_cond if hasattr(config.unet, 'class_cond') else False,
                      num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None
                      )

    model.to(dev)
    
    FM = EquilibriumMatching(sched=config.FM.sched, lamda=config.FM.lamda, a=0.8)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
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
                class_cond=config.unet.class_cond if hasattr(config.unet, 'class_cond') else False)

if __name__ == '__main__':
    main(sys.argv[1])