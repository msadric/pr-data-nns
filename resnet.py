import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        logits = out
        probas = F.softmax(logits, dim=1)
        return logits, probas


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

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
                
        # Validation
        """
        model.eval()
        
        with torch.no_grad(): 
            
            right_train, num_train = get_right_ddp(model, train_loader)
            right_valid, num_valid = get_right_ddp(model, valid_loader)
            
            dist.all_reduce(right_train.clone().detach(), op=dist.ReduceOp.SUM)
            dist.all_reduce(right_valid.clone().detach(), op=dist.ReduceOp.SUM)
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
         """   
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

def main():
    
    world_size = int(os.environ["SLURM_NTASKS"], 2)
    rank = int(os.environ["SLURM_PROCID"], 0)

    
    print("Rank, world_size, device_count:", rank, world_size, torch.cuda.device_count())
    
    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")  # Get master node address from SLURM environment variable.
    if address is None:
        raise ValueError("SLURM_LAUNCH_NODE_IPADDR is not set.")  
        
    port = 1234 # Set port number.
    os.environ["MASTER_ADDR"] = str(address) # Set master node address.
    os.environ["MASTER_PORT"] = str(port) # Set master node port.
    
    # Use NCCL backend for communication as GPU training
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    
    # Check if process group has been initialized successfully (+ backend used)
    if torch.distributed.is_initialized():
        print("Successfully initialized process group: {}".format(torch.distributed.get_backend()))
    else:
        print("Failed to initialize process group: {}".format(torch.distributed.get_backend()))
    
    
    seed = 1
    batch_size    = 256 
    num_epochs    = 20 

    set_all_seeds(seed=seed)
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),        
        torchvision.transforms.CenterCrop((64, 64)),            
        torchvision.transforms.ToTensor(),                
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_loader, valid_loader = get_dataloaders_mnist(batch_size=batch_size,
                                                                      train_transforms=train_transforms,
                                                                      test_transforms=test_transforms)
    
    if rank==0:
        test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
    model = ResNet18(num_classes=10)
    model = model.cuda()
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)  

    loss_list, train_acc_list, valid_acc_list = [], [], []
    loss_list, train_acc_list, valid_acc_list = train_model(model=model, 
                                                        num_epochs=num_epochs,
                                                        train_loader=train_loader, 
                                                        valid_loader=valid_loader, 
                                                        optimizer=optimizer,)
    
    if rank == 0:
        test_acc = compute_accuracy(model, test_loader)
        print(f'Test accuracy: {test_acc :.2f}%')
        
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()