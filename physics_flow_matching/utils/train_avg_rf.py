import torch as th
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from physics_flow_matching.utils.pre_procs_data import get_batch_avg

def restart_func(restart_epoch, path, model, optimizer, sched=None):
    assert restart_epoch != None, "restart epoch not initialized!"
    print(f"Loading state from checkpoint epoch : {restart_epoch}")
    state_dict = th.load(f'{path}/checkpoint_{restart_epoch}.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    start_epoch = restart_epoch + 1
    
    if 'sched_state_dict' in state_dict.keys():
        sched.load_state_dict(state_dict['sched_state_dict']) 
        
    return start_epoch, model, optimizer, sched


def train_model(model: nn.Module, FM, train_dataloader,
                optimizer: optim.Optimizer, sched: optim.Optimizer|None, loss_fn, writer : SummaryWriter,
                num_epochs, print_epoch_int,
                save_epoch_int, print_within_epoch_int, path,
                device,
                restart=False, restart_epoch=None):
    
    if restart:
        start_epoch, model, optimizer, sched = restart_func(restart_epoch, path, model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
        
    for epoch in range(start_epoch, num_epochs):
        model.train()
        iter_val = 0
        epoch_loss = 0.
        
        for iteration, info in enumerate(train_dataloader):
            x0, x1 = info

            t, r, xt, vt = get_batch_avg(FM, x0.to(device), x1.to(device))

            u_pred, du_dt = th.autograd.functional.jvp(model, inputs=(r, t, xt), v=(th.zeros_like(r).to(device), th.ones_like(t).to(device), vt), create_graph=True)
            
            u = ((r - t) * du_dt + vt).detach()
            
            loss = loss_fn(u_pred, u)
                
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()      
            iter_val += 1
            epoch_loss += loss.item()

            if iteration % print_within_epoch_int == 0:
                writer.add_scalar("Within_Epoch/train_loss", loss.item(), iteration)
                print(f"----Train Epoch {epoch}, Iter loss at {iteration}: {loss.item()}")
        
        epoch_loss /= iter_val
        
        if sched is not None:
             sched.step()
        
        if epoch % print_epoch_int == 0:
            print(f"Avg Train Loss at Epoch {epoch} : {epoch_loss}")
            writer.add_scalar("Epoch/train_loss", epoch_loss, epoch)
            
        if (epoch % save_epoch_int) == 0 or (epoch == (num_epochs - 1)):
            print(f"Saving model details at epoch: {epoch}")
            if sched is not None:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': sched.state_dict()
                            }, f'{path}/checkpoint_{epoch}.pth')
            else:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f'{path}/checkpoint_{epoch}.pth')
        
    writer.close()
    print("Training Complete!")
    return 0