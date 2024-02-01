
def get_dataloaders_mnist(batch_size,
                          num_workers=0,
                           root='data',
                           validation_fraction=0.1,
                           train_transforms=None,
                           test_transforms=None):

    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    # Load training data.
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=train_transforms,
        download=True
    )

    # Load validation data.
    valid_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
    
    # Get samplers.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=torch.distributed.get_world_size(), 
        rank=torch.distributed.get_rank(), 
        shuffle=True, 
        drop_last=True
    )
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, 
        num_replicas=torch.distributed.get_world_size(), 
        rank=torch.distributed.get_rank(), 
        shuffle=True, 
        drop_last=True)
    

    # Get data loaders.
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        sampler=train_sampler
    )

    return train_loader, valid_loader

def set_all_seeds(seed): # exclude nondeterminism
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(model, data_loader):

    with torch.no_grad():

        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_samples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.cuda() 
            
            result = model(features)
            _  , predictions = torch.max(result[0], dim=1)
            num_samples += targets.size(0)
            correct_pred += (predictions == targets).sum()
    
    return correct_pred.float() / num_samples * 100 

def get_right_ddp(model, data_loader):

    with torch.no_grad(): 

        correct_pred, num_samples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.cuda()
            targets = targets.cuda()
            
            result = model(features)
            _, predictions = torch.max(result[0], dim=1)
            num_samples += targets.size(0)
            correct_pred += (predictions == targets).sum()
        
    return correct_pred, num_samples      
    
def train_model(model, num_epochs, train_loader,
                valid_loader, optimizer):

    loss_fn = F.cross_entropy
    start = time.time()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    if rank == 0:
        loss_history, train_acc_history, valid_acc_history = [], [], []

    
    for epoch in range(num_epochs): 
        
        train_loader.sampler.set_epoch(epoch) 
      
        # Training
        
        model.train()
        for batch_idx, (features, targets) in enumeion=0.1,
                           train_transforms=op=dist.ReduceOp.SUM) # Sum up mini-mini-batch losses from all processes.
            loss /= world_size # Divide by number of processes to get average.
            
            optimizer.step()            
    
            if rank == 0:
                loss_history.append(loss.item())
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
                
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
            
            # Need to think about this TODO
            
            train_acc = right_train / num_train * 100
            valid_acc = right_valid / num_valid * 100

            #train_acc = compute_accuracy(model, train_loader)
            #valid_acc = compute_accuracy(model, valid_loader)
            
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
            print(f'Total Training Time: {elapsed:.2f}s')
    
    return loss_history, train_acc_history, valid_acc_history
