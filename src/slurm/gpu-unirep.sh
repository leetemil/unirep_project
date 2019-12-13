#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
# we run on the gpu partition and we allocate 4 titan x
#SBATCH -p gpu --gres=gpu:titanx:4
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0:05:00

# Info:
hostname
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
echo "CUDA version:"
nvcc --version

# Script:
# -u: Unbuffered output
python3 -u main.py

# End
echo "Finished"
