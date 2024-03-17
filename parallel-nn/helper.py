import os
import torch
import random
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep learning model.')
    
    # Model choices
    parser.add_argument('--model', choices=['ResNet', 'AlexNet'], default='ResNet',
                        help='Choose the deep learning model (ResNet, AlexNet)')

    # Dataset choices
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR-10'], default='MNIST',
                        help='Choose the dataset (MNIST, CIFAR-10)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--results_folder', type=str, default='results', help='Path to the results folder')

    args = parser.parse_args()
    return args

def set_all_seeds(seed):
    """
    Set the random seeds for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(model, data_loader):
    """
    Compute the accuracy of a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The accuracy of the model on the data loader.
    """
    with torch.no_grad():

        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_samples = 0, 0

        for _, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.cuda() 
            
            result = model(features)
            
            _  , predictions = torch.max(result[0], dim=1)
            num_samples += targets.size(0)
            correct_pred += (predictions == targets).sum()
    
    return correct_pred.float() / num_samples * 100 


def computer_accuracy_ddp(model, data_loader):
    """
    Compute the accuracy of a model on a given data loader using Distributed Data Paralmodel_arg,dataset_arg,seed_arg,batch_size_arg,num_epoch_arg,num_devices,training_time,train_acc_list[-1],valid_acc_list[-1],test_acc
lel (DDP).

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The accuracy of the model on the data loader.
    """
    correct_pred, num_samples = get_right_ddp(model, data_loader)
    return correct_pred.item() / num_samples.item() * 100


def get_right_ddp(model, data_loader):
    """
    Get the number of correctly predicted samples and the total number of samples using Distributed Data Parallel (DDP).

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader.

    Returns:
        Tuple[int, int]: A tuple containing the number of correctly predicted samples and the total number of samples.
    """
    with torch.no_grad(): 

        correct_pred, num_samples = 0, 0

        for _, (features, targets) in enumerate(data_loader):

            features = features.cuda()
            targets = targets.cuda()
            
            result = model(features)
            _, predictions = torch.max(result[0], dim=1)
            num_samples += targets.size(0)
            correct_pred += (predictions == targets).sum()
        
    return correct_pred, num_samples