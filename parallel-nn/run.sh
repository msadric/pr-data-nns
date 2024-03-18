#!/bin/bash

#SBATCH --job-name=16_gpu_test

#SBATCH --partition=gpu_8

#SBATCH --gres=gpu:8 # number of requested GPUs (GPU nodes shared between multiple jobs)
#SBATCH --ntasks=16

#SBATCH --ntasks-per-gpu=1

#SBATCH --time=03:00:00 # wall-clock time limit
#SBATCH --mem=32000
#SBATCH --nodes=2
#SBATCH --mail-type=all
#SBATCH --mail-user=uunsi@student.kit.edu

module purge # Unload currently loaded modules.

source "parallel-venv/bin/activate"     

srun python -u main.py --batch_size 256 --dataset MNIST --model ResNet



















