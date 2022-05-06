# EME
EME (Energy Measurements with EAR) for NQS (Neural Quantum States) use case. 

The 3 main scripts are:

1. "multi_cpu_normal.sh" Submits mulitple (60) jobs on the cpu with the cpu normal method. Runs one network on the cpu (dual socket) at a time and devides created thread over the cores. It only varies alpha (the hidden layer density) and thus the network size between jobs. Command: `source multi_cpu_normal.sh`.

2. "multi_cpu_parallel.sh" Submits multiple (60) jobs on the cpu with the cpu parallel method. Runs 1 network on one core and thus 40 networks in parallel on all 40 available cores. It only varies alpha (the hidden layer density) and thus the network size between jobs. Command: `source multi_cpu_parallel.sh`.

3. "multi_gpu.sh" Submits mulitple (60) jobs on the gpu with the gpu method. Runs one network on the gpu at a time. It only varies alpha (the hidden layer density) and thus the network size between jobs. Command: `source multi_gpu.sh`.

After one of these 3 scripts is done running the data is collected with: `
source get_results.sh joblist/cpu_normal_method.csv` or `source get_results.sh joblist/cpu_parallel_method.csv` or `source get_results.sh joblist/gpu_method.csv` for the three respective methods.

The trained Models are in the folder optimized_W and the ED_amplitude.csv file is the numerical exact solution for a 4x4 2D square lattice. 