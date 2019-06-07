#!/bin/bash
# @ job_name= multilayer
# @ initialdir= .
# @ output= multilayer-%j.out
# @ error= %j.err
# @ total_tasks= 1
# @ gpus_per_node= 4
# @ cpus_per_task= 1
# @ features= k80
# @ wall_clock_limit= 00:23:30

# #SBATCH --reservation=PATC_DL2

#::Code for running in MT with Python 2.7.12
module purge
module load merovingian
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
python run_multilayer.py
#python /gpfs/home//nct01/nct01022/multiGPUproof.py
