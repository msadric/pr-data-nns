import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ResNet import ResNet18
from helper import * 

def main():
    
    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])

    
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
    num_epochs    = 5

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
    
    train_loader, valid_loader = get_dataloaders_mnist(batch_size=batch_size, train_transforms=train_transforms, test_transforms=test_transforms)
    
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