import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from torch.utils.data import DataLoader
# from physics_flow_matching.multi_fidelity.synthetic.dataset import flow_guidance_dists
from physics_flow_matching.multi_fidelity.synthetic.dataset import Syn_Data_FM
from physics_flow_matching.utils.train_mnist import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher #FlowMatcher
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
        
    model = UNetModel(dim=config.unet.dim, num_channels=config.unet.num_channels, num_res_blocks=config.unet.num_res_blocks)

    model.to(dev)

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=config.FM.sigma)
    
    # FM = FlowMatcher(sigma=config.FM.sigma,
    #                  add_heavy_noise=config.FM.add_heavy_noise if hasattr(config.FM, 'add_heavy_noise') else False,
    #                   nu=config.FM.nu if hasattr(config.FM, 'nu') else th.inf)
    
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