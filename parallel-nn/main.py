import torch
import torch.nn.functional as F
import torchvision
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from models.ResNet import ResNet18
from models.AlexNet import AlexNet
from helper import set_all_seeds, compute_accuracy, parse_arguments
from data import get_dataloaders
from train import train_model
import numpy as np
import csv
from mplot import plot_losses
import random


# This is made for running on bwUniCluster 2.0 
def main():
    args = parse_arguments()
    model_arg, dataset_arg, seed_arg, batch_size_arg, num_epoch_arg, results_folder_arg = args.model, args.dataset, args.seed, args.batch_size, args.epochs, args.results_folder
    
    world_size = int(os.getenv("SLURM_NPROCS"))
    rank = int(os.environ["SLURM_PROCID"])
    num_devices_cuda = torch.cuda.device_count()
    num_devices = torch.tensor(num_devices_cuda).cpu().item()

    print("Rank, world_size, device_count:", rank, world_size, torch.cuda.device_count())

    
    results_path = os.path.join(results_folder_arg, f'{dataset_arg}_{model_arg}_{seed_arg}_{batch_size_arg}_{num_epoch_arg}_{world_size}_{num_devices}')
    
    if rank==0:
        if not os.path.exists(results_path):
            os.makedirs(results_path)

    
    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")  # Get master node address from SLURM environment variable.
    if address is None:
        raise ValueError("SLURM_LAUNCH_NODE_IPADDR is not set.")  

    print(address)
    port = 1234 + world_size + num_devices + batch_size_arg # unique port, when multiple in queue
    print(port)
    os.environ["MASTER_ADDR"] = str(address) # Set master node address.
    os.environ["MASTER_PORT"] = str(port) # Set master node port.
        
    # Use NCCL backend for communication as GPU training
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    
    # Check if process group has been initialized successfully (+ backend used)
    if torch.distributed.is_initialized():
        print("Successfully initialized process group: {}".format(torch.distributed.get_backend()))
    else:
        print("Failed to initialize process group: {}".format(torch.distributed.get_backend()))


    seed = seed_arg
    batch_size    = batch_size_arg
    num_epochs    = num_epoch_arg

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
    
    model = ResNet18(num_classes=10, input_channels=1)


    train_loader, valid_loader = get_dataloaders(batch_size=batch_size, dataset_type='mnist')
        
    if rank==0:
        test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
    if dataset_arg == 'CIFAR-10':
        model = ResNet18(num_classes=10, input_channels=3)
        train_loader, valid_loader = get_dataloaders(batch_size=batch_size, dataset_type='cifar10')
        
        # Transforms applied to training data (randomness to make network more robust against overfitting)
        train_transforms = (
            torchvision.transforms.Compose(  # Compose several transforms together.
                [
                    torchvision.transforms.Resize(
                        (70, 70)
                    ),  # Upsample CIFAR-10 images to make them work with AlexNet.
                    torchvision.transforms.RandomCrop(
                        (64, 64)
                    ),  # Randomly crop image to make NN more robust against overfitting.
                    torchvision.transforms.ToTensor(),  # Convert image into torch tensor.
                    torchvision.transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),  # Normalize to [-1,1] via (image-mean)/std.
                ]
            )
        )
    
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((70, 70)),
                torchvision.transforms.CenterCrop((64, 64)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if rank==0:
            test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            
    elif dataset_arg == 'ImageNet':
        model = ResNet18(num_classes=1000, input_channels=3)
        train_loader, valid_loader = get_dataloaders(batch_size=batch_size, dataset_type='imagenet')
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        if rank==0:
            test_dataset = torchvision.datasets.ImageNet(root='data', train=False, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            
    if model_arg == 'AlexNet':
        print("right model choosen")
        model = AlexNet(num_classes=10, input_channels=1)
        if dataset_arg == 'CIFAR-10':
            model = AlexNet(num_classes=10, input_channels=3)
        elif dataset_arg == 'ImageNet':
            model = AlexNet(num_classes=1000, input_channels=3)
    
    model = model.cuda()
    
    ddp_model = DDP(model)
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)  

    loss_list, train_acc_list, valid_acc_list = [], [], []
    loss_list, train_acc_list, valid_acc_list, training_time = train_model(model=model, 
                                                        num_epochs=num_epochs,
                                                        train_loader=train_loader, 
                                                        valid_loader=valid_loader, 
                                                        optimizer=optimizer)
    
    if rank == 0:
        # Save loss and accuracy history.
        torch.save(loss_list, os.path.join(results_path, 'loss.npy'))
        torch.save(train_acc_list, os.path.join(results_path, 'train_acc.npy'))
        torch.save(valid_acc_list, os.path.join(results_path, 'valid_acc.npy'))
        torch.save(ddp_model.state_dict(), os.path.join(results_path, 'ddp_trained_model.pth'))


        # Plot
        plot_losses(loss_list, train_acc_list, valid_acc_list, results_path, num_epochs, batch_size)
        
        test_acc = compute_accuracy(model, test_loader)
        print(f'Test accuracy: {test_acc :.2f}%')
    

        # Your summary information
        summary = [model_arg, dataset_arg, seed_arg, batch_size_arg, num_epoch_arg, world_size, training_time, train_acc_list[-1].item(), valid_acc_list[-1].item(), test_acc.item()]
        
        # Convert each element to string and join with commas
        summary_string = ','.join(map(str, summary))
        
        file_path = 'results/summary.csv'
        
        # Check if the file already exists
        file_exists = os.path.isfile(file_path)
        
        # If the file doesn't exist, create a new file with the header
        if not file_exists:
            header = ["Model", "Dataset", "Seed", "Batch Size", "Num Epoch", "Num Devices", "Training Time", "Train Accuracy", "Valid Accuracy", "Test Accuracy"]
        
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        # Append the summary row to the CSV file
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summary_string.split(','))
            
    
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()