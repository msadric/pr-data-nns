#!/bin/bash

# Source for this script is the Lecture Scalable AI in WT 2023/24 at KIT.

#SBATCH --job-name=mnist_seq                   # job name
#SBATCH --partition=multiple               # queue for resource allocation
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --time=5:00                        # wall-clock time limit
#SBATCH --mem=90000                        # memory per node 
#SBATCH --cpus-per-task=1                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

export IBV_FORK_SAFE=1
export VENVDIR=<path/to/your/venv/folder>  # Export path to your virtual environment.
export PYDIR=<path/to/your/python/script>  # Export path to directory containing Python script.

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/gnu/10.2              # Load required modules.
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2
module load lib/hdf5/1.12.2-gnu-10.2-openmpi-4.1

source ${VENVDIR}/bin/activate              # Activate your virtual environment.

mpirun --mca mpi_warn_on_fork 0 python -u ${PYDIR}/cdist.py