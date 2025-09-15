def finetune_network(model):
    
    ## exp 5
    
    for param in model.parameters():
        param.requires_grad_(False)
        
    for i in range(3, 0, -1): 
        model.output_blocks[-i].requires_grad_(True)
    
    model.out.requires_grad_(True) 
    
    return model
    
    ## exp 6
    
    # return model # allow gradients throughtout model but only update the first few layers using the optimizer    
