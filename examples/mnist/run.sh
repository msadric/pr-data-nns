#!/bin/bash

#SBATCH --job-name=mnist_1_cpu             # job name
#SBATCH --output=mnist_1_cpu.out           # output file
#SBATCH --partition=multiple               # queue for the resource allocation.
#SBATCH --time=10:00                       # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uunsi@student.kit.edu  # notification email address

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VENVDIR="/pfs/data5/home/kit/stud/uunsi/parallel-venv" # Export path to your virtual environment.
export PYDIR="/pfs/data5/home/kit/stud/uunsi/mnist" # Export path to directory containing Python script.

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2
module load lib/hdf5/1.12.2-gnu-10.2-openmpi-4.1  

source ${VENVDIR}/bin/activate # Activate your virtual environment.

mpirun -n 1 python ${PYDIR}/main.py # Run your Python script in parallel.
