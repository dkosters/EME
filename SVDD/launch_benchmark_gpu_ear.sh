#!/bin/sh
mkdir -p joblist

SVDDname=$1
zdim=$2
layers=$3
ft=$4
precision=$5

device='gpu'
Jobname="${SVDDname}_${precision}_${device}"
rm joblist/$Jobname.csv


echo "jobid,batch size,iterations,id,device" >> joblist/$Jobname.csv

iterations=2
batchsize=1
for i in {1..20}  #repeat x amount of times if node is not specified this will alter between two possible nodes  
do 
	jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
    echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
	echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
done

iterations=10
batchsize=10
for i in {1..1}  #repeat x amount of times if node is not specified this will alter between two possible nodes  
do 
	jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
    echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
	echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
done

iterations=20
batchsize=100
for i in {1..1}  #repeat x amount of times if node is not specified this will alter between two possible nodes  
do 
	jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
    echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
	echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
done


iterations=150
for batchsize in 1000 10000 100000
do
	for i in {1..1}  #repeat x amount of times if node is not specified this will alter between two possible nodes
	do
	    jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
        echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
		echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
	done
done

iterations=150
for batchsize in 1000000
do
	for i in {1..10}  #repeat x amount of times if node is not specified this will alter between two possible nodes
	do
	    jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
        echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
		echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
	done
done

iterations=250
for batchsize in 5000000 10000000 50000000
do
	for i in {1..1}  #repeat x amount of times if node is not specified this will alter between two possible nodes
	do
	    jobid=`sbatch --parsable benchmarkscripts/svdd_gpu_ear.sh ${zdim} "${layers}" ${ft} ${iterations} ${batchsize} ${SVDDname}.h5 ${precision}`  # first dim then hidden layers of size then fixed target then iteration
        echo "$jobid,$batchsize,$iterations,${SVDDname},$device" >> joblist/$Jobname.csv
		echo "submitting job $jobid with batchsize $batchsize and $iterations iterations using id ${SVDDname} on $device"
	done
done

echo "check if they are on squeue"

squeue -u bryan_esc





