import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from helper import get_right_ddp
from torch.optim import lr_scheduler

def train_model(model, num_epochs, train_loader,
                valid_loader, optimizer):

    loss_fn = F.cross_entropy
    start = time.time()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    if rank == 0:
        loss_history, train_acc_history, valid_acc_history = [], [], []

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    
    for epoch in range(num_epochs): 
        
        train_loader.sampler.set_epoch(epoch) 

        # Training
        model.train()
        
        
        for batch_idx, (features, targets) in enumerate(train_loader): # Loop over mini batches.
            
            features = features.cuda()
            targets = targets.cuda()
            
            result = model(features) # Forward pass
            loss = loss_fn(result[0], targets) # Compute loss
            optimizer.zero_grad() # Zero out gradients from previous step, because PyTorch accumulates them
            loss.backward() # Backward pass
            
            dist.all_reduce(loss.clone().detach(), op=dist.ReduceOp.SUM) # Sum up mini-mini-batch losses from all processes.
            loss /= world_size # Divide by number of processes to get average.
            
            optimizer.step()
    
            if rank == 0:
                loss_history.append(loss.item())
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
                
        scheduler.step() # Update the learning rate at the end of each epoch
        
        # Validation
        
        model.eval()
        
        with torch.no_grad(): 
            
            right_train, num_train = get_right_ddp(model, train_loader)
            right_valid, num_valid = get_right_ddp(model, valid_loader)
            
            dist.all_reduce(right_train.clone().detach(), op=dist.ReduceOp.SUM)
            dist.all_reduce(right_valid.clone().detach(), op=dist.ReduceOp.SUM)
            
            num_train = torch.tensor(num_train).cuda()
            num_valid = torch.tensor(num_valid).cuda()
            
            dist.all_reduce(num_train.clone().detach(), op=dist.ReduceOp.SUM)
            dist.all_reduce(num_valid.clone().detach(), op=dist.ReduceOp.SUM)
                        
            train_acc = right_train / num_train * 100
            valid_acc = right_valid / num_valid * 100
            
            if rank == 0:
                  print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
                  valid_acc_history.append(valid_acc)
                  train_acc_history.append(train_acc)
           
        elapsed = time.time() - start
        Elapsed = torch.Tensor([elapsed]).cuda()
        
        dist.all_reduce(Elapsed, op=dist.ReduceOp.SUM)
        elapsed = Elapsed.item() / world_size
        
        if rank == 0:
            print(f'Total Training Time: {elapsed:.2f}s')
    
    # Total Training Time        
    elapsed = time.time() - start
    Elapsed = torch.Tensor([elapsed]).cuda()
    dist.all_reduce(Elapsed, op=dist.ReduceOp.SUM)
    elapsed = Elapsed.item() / world_size
    
    if rank == 0:
        training_time = elapsed
        print(f'Total Training Time: {elapsed:.2f}s')
        
    
    return (None if rank != 0 else (loss_history, train_acc_history, valid_acc_history, training_time))
