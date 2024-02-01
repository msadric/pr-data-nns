#!/bin/bash

#SBATCH --job-name=alex1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1   # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --time=2:00:00 # wall-clock time limit, adapt if necessary
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u????@student.kit.edu

module purge                                        # Unload currently loaded modules.
module load compiler/gnu/10.2
module load devel/cuda/10.2

source <path to your venv folder>/bin/activate      # Activate your virtual environment.

python -u <path to your python script>/alex.py