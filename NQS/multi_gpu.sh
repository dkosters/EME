#!/bin/sh

# load ear 
module load ear 

# makes a director where you save job information in a csv file 
mkdir -p joblist

# example measurment 3 for gpu mehtod 
device='gpu'
Jobname="gpu_method"
dtype="fp32"

echo "jobid,iterations,alpha,device,dtype" >> joblist/$Jobname.csv

iterations=100000 # first argument can be something like the number of iterations such as in this example
for alpha in 1 2 4 6 8 10
do
    for i in {1..10}  # repeat x amount of times if node is not specified this will alter between two possible nodes  
    do 
        jobid=`sbatch --parsable gpu.sh ${alpha} ${iterations} ${dtype}`  # single job submission for the gpu method
        echo "$jobid,$iterations,$alpha,$device,$dtype" >> joblist/$Jobname.csv  # pushes job input to a csv file 
        echo "submitting job $jobid and $iterations iterations on $device"  # prints some general information 
    done
done 