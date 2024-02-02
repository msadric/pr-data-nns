import torch
import torchvision
import numpy as np

def get_dataloaders(batch_size, num_workers=0, root='data', validation_fraction=0.1, dataset_type='mnist'):
    """
    Get data loaders for various datasets.

    Args:
        batch_size (int): The batch size for the data loaders.
        num_workers (int, optional): The number of worker threads to use for data loading. Defaults to 0.
        root (str, optional): The root directory where the dataset will be saved. Defaults to 'data'.
        validation_fraction (float, optional): The fraction of training data to use for validation. Defaults to 0.1.
        dataset_type (str, optional): Type of dataset ('mnist', 'cifar10', 'imagenet'). Defaults to 'mnist'.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    
    if dataset_type == 'mnist':
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

    elif dataset_type == 'cifar10':
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load training data.
        train_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            transform=train_transforms,
            download=True
        )

        # Load validation data.
        valid_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            transform=test_transforms
        )

    elif dataset_type == 'imagenet':
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

        # Load training data.
        train_dataset = torchvision.datasets.ImageNet(
            root=root,
            split='train',
            transform=train_transforms,
            download=True
        )

        # Load validation data.
        valid_dataset = torchvision.datasets.ImageNet(
            root=root,
            split='val',
            transform=test_transforms,
            download=True
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
        drop_last=True
    )

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
