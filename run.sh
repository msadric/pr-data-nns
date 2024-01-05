#!/bin/bash

#SBATCH --job-name=resnet
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:2 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=2
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=45:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uunsi@student.kit.edu

module purge                                        # Unload currently loaded modules.
#module load compiler/gnu/10.2
module jupyter/tensorflow/2023-10-10
#module load devel/cuda/10.2  

source "martin/bin/activate"      

unset SLURM_NTASKS_PER_TRES

srun python -u resnet.py