#!/bin/sh

# load ear 
module load ear 

# makes a director where you save job information in a csv file 
mkdir -p joblist

# example measurement 1 for CPU normal method 
device='cpu'
Jobname="Name that refers to your batch of job submissions for cpu method"

echo "jobid,iterations,layersize,device" >> joblist/$Jobname.csv

iterations=100 # first argument can be something like the number of iterations such as in this example
for layer in 10 100 1000 10000
do
    for i in {1..10}  # repeat x amount of times if node is not specified this will alter between two possible nodes  
    do 
        jobid=`sbatch --parsable cpu_normal.sh ${iterations} ${layer}`  # single job submission 
        echo "$jobid,$iterations,$layer,$device" >> joblist/$Jobname.csv  # pushes job input to a csv file 
        echo "submitting job $jobid and $iterations iterations on $device"  # prints some general information 
    done
done 

# example measurement 2 for CPU parallel method 
device='cpu'
Jobname="Name that refers to your batch of job submissions for cpu parlalel method"

echo "jobid,iterations,layersize,device" >> joblist/$Jobname.csv

iterations=100 # first argument can be something like the number of iterations such as in this example
for layer in 10 100 1000 10000
do
    for i in {1..10}  # repeat x amount of times if node is not specified this will alter between two possible nodes  
    do 
        jobid=`sbatch --parsable cpu_parallel.sh ${iterations} ${layer}`  # single job submission 
        echo "$jobid,$iterations,$layer,$device" >> joblist/$Jobname.csv  # pushes job input to a csv file 
        echo "submitting job $jobid and $iterations iterations on $device"  # prints some general information 
    done
done 

# example measurment 3 for gpu mehtod 
device='gpu'
Jobname="Name that refers to your batch of job submissions for gpu method"

echo "jobid,iterations,layersize,device" >> joblist/$Jobname.csv

iterations=100 # first argument can be something like the number of iterations such as in this example
for layer in 10 100 1000 10000
do
    for i in {1..10}  # repeat x amount of times if node is not specified this will alter between two possible nodes  
    do 
        jobid=`sbatch --parsable gpu.sh ${iterations} ${layer}`  # single job submission 
        echo "$jobid,$iterations,$layer,$device" >> joblist/$Jobname.csv  # pushes job input to a csv file 
        echo "submitting job $jobid and $iterations iterations on $device"  # prints some general information 
    done
done 