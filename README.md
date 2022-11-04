# EME
EME (Energy Measurements with EAR)

This is a guide to perform energy, latency and throughput measurements for inference tasks on the ESC cluster of SURF using EAR. 

First results of a condensed matter and particle physics use case can be found on [arXiv:2209.10481](https://arxiv.org/pdf/2209.10481.pdf)

It's assumed you are using python for your network and that you can adjust parameters from command line arguments such as network size, device and number of inference iterations. 

First step is to setup your environment.
This can be done with the commands: 
~~~
python3 -m venv venv_name 
source venv_name/bin/activate 
python -m pip install -r requirements.txt 
~~~
You still need to install the libraries where your python script is dependent on, such as PyTorch or tensorflow. 

If you are done installing all the libraries you can deactivate the environment with:
~~~
deactivate
~~~

There are three main shell scripts that submit a job on the CPU and GPU, namely, cpu_normal.sh, cpu_parallel.sh and gpu.sh 

cpu_normal.sh runs your single python script on the cpu.
cpu_parallel.sh runs one python script per core of the cpu. 
gpu.sh runs your single python script on the gpu. 

You will need to edit these shell script such that they refer to your python script. 

Change the environment path in the shell scripts to your environment path. That is, change PathToYourVirtualEnv to the correct path in the following line (this is present in all three scripts): 

~~~
source $HOME/PathToYourVirtualEnv/bin/activate 
~~~

Next change PathToYourPythonScript and PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole to the path of your python script and the path where you want to store console output of the script respectivily. Altehrnativly you can also remove 
~~~
>> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt
~~~
to not save the console output. Also change the arguments to your desired arguments. There should still be an argument that points to the correct device or you should make sure form within your script you have defined the correct device. 

For cpu_normal.sh change: 
~~~
srun --ear=on python $HOME/PathToYourPythonScript.py --argument $1 --argument $2 --argument $3 --device "cpu"  >> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt &
~~~

For cpu_parallel.sh change:
~~~
numactl --physcpubind=$i python $HOME/PathToYourPythonScript.py --argument $1 --argument $2 --argument $3 --device "cpu"  >> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt &
~~~

For gpu.sh change:
~~~
srun --ear=on python $HOME/PathToYourPythonScript.py --argument $1 --argument $2 --argument $3 --device "gpu"  >> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt &
~~~

In the measurement.sh file you find 3 example measurements (one for cpu normal method one for cpu parallel method and one for gpu method). Each measurement submits a batch of jobs. In this case the jobs have varying layer sizes (10-10000) and all have 100 iterations. 

Eech example measurement makes an csv file where the input data of the jobs will be stored. This is necessary to later quickly gather all the data measured by EAR. 

You can use this measurement.sh script as a blueprint for your own measurement. Just make sure that you make a csv file the same way as in the script and save the jobid in it per job, other information is up to whatever you want to save you can. 

Finally all the information and measurement results can be gathered in a single csv file by running get_results.sh:
~~~
source get_results.sh joblist/your_batch_job_file.csv
~~~
To make get_results.sh work you again have to change the environment within the shell script to your environment. 

There will now be 2 files in a folder called earoutput which contain all the information of your measurement. 

