#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
# we run on the gpu partition and we allocate 1 titan x
#SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=17:00:00

# Info:
hostname
echo $CUDA_VISIBLE_DEVICES

# Script:
pip3 install --user torch torchvision
pip3 install --user biopython
python3 -u main.py

# End
echo "Finished"
