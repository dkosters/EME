#!/bin/sh

module load ear

# activate environment 
source $HOME/PathToYourVirtualEnv/bin/activate 

python3 data_processing.py --filepath $1 # 1 is the path file to your measurents id batch size etc. 

deactivate
