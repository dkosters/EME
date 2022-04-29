#!/bin/sh 
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --ear-verbose=1
#SBATCH --ear-policy=monitoring 
#SBATCH --ear=on

# making sure the threads are bind to a core (preventing thread migration)
# also bind the threads to the cores in order i.e. first core 1 then core 2 ...
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#load libraries
module load 2020

# activate environment 
source $HOME/NQS/nqs_env/bin/activate 

# makes sure only CPU is used 
export CUDA_VISIBLE_DEVICES=''

# used to get advanced metrics from ear 
export SLURM_LOADER_LOAD_NO_MPI_LIB=python

srun --ear=on python $HOME/NQS/rbm.py --alpha $1 --iterations $2 --device "cpu" &

wait
