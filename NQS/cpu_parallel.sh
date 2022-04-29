#!/bin/sh 
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --ear-verbose=1
#SBATCH --ear-policy=monitoring 
#SBATCH --ear=on

#load numactl
module load 2020
module load numactl/2.0.13-GCCcore-10.2.0

# activate virtual environment 
source $HOME/NQS/nqs_venv/bin/activate 

# makes sure only CPU is used 
export CUDA_VISIBLE_DEVICES=''

# for loop 0-39 since there are 40 cores 
for i in {0..39}
do
    numactl --physcpubind=$i python $HOME/NQS/rbm.py --alpha $1 --iterations $2 --device "cpu" &
done

wait
