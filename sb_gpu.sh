#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name="multiGPU"
#SBATCH --workdir=.
#SBATCH --output=multigpu_%j.out
#SBATCH --error=multigpu_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=00:03:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

#python multi_gpu_test.py 2
python run_multilayer_gpu.py 1

