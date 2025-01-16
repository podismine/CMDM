#!/bin/bash

#SBATCH --partition=gpu #Name of your job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G #Number of cores to reserve
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1 #Number of cores to reserve
#SBATCH --output=run_logs/DM_%x.o%j   #Path to file for the STDOUT standard output
#SBATCH --error=run_logs/DM_%x.e%j    #Path to file for the STDERR error output


source /home/yyang/miniconda3/bin/activate

cuda_visible_devices=$1 python 08_train_best.py -m $2; cuda_visible_devices=$1 python 07_eval.py -m $2