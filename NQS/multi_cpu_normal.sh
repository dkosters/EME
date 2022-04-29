#!/bin/sh

# load ear 
module load ear 

# makes a director where you save job information in a csv file 
mkdir -p joblist

# example measurement 1 for CPU normal method 
device='cpu'
Jobname="cpu_normal_method"

echo "jobid,iterations,alpha,device" >> joblist/$Jobname.csv

iterations=100000  # number of iterations
for alpha in 1 2 4 6 8 10  # iterate over the hidden layer density 
do
    for i in {1..10}  # repeat 10 amount of times if node is not specified this will alter between two possible nodes  
    do 
        jobid=`sbatch --parsable cpu_normal.sh ${alpha} ${iterations}`  # single job submission with the cpu normal method  
        echo "$jobid,$iterations,$alpha,$device" >> joblist/$Jobname.csv  # pushes job input to a csv file 
        echo "submitting job $jobid and $iterations iterations on $device"  # prints some general information 
    done
done 