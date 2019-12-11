#!/bin/bash
#SBATCH --job-name=DummySaveFileTest
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-00:02:00
#Skipping many options! see man sbatch

echo Hello world!
echo file content > dummyfile.txt
