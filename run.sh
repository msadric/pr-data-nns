#!/bin/bash

#SBATCH --job-name=resnet_with_purge
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=45:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uunsi@student.kit.edu

module purge                                        # Unload currently loaded modules.

module jupyter/tensorflow/2023-10-10
module load devel/cuda/11.8  

source "8-1-venv/bin/activate"      

srun python -u resnet.py