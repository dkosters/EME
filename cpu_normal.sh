#!/bin/sh 
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --ear-verbose=1
#SBATCH --ear-policy=monitoring 
#SBATCH --ear=on

#load libraries
module load 2020

# activate environment 
source $HOME/PathToYourVirtualEnv/bin/activate 

# makes sure only CPU is used 
export CUDA_VISIBLE_DEVICES=''

# used to get advanced metrics from ear 
export SLURM_LOADER_LOAD_NO_MPI_LIB=python

srun --ear=on python $HOME/PathToYourPythonScript.py --argument $1 --argument $2 --argument $3 --device "cpu"  >> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt &

wait
