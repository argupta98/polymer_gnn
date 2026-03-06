#!/bin/bash
#SBATCH --job-name=gnn
#SBATCH -p gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --constraint='L40'
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err

mkdir -p logs

source ~/.bashrc
conda activate GLAMOUR

python hyperparameter_optimization.py --seed 103 --GPU 0 --model MPNN --num_epochs 200 --num_trials 100 --rand_samples 15


### CALL SCRIPT USING: sbatch run_hpo.sh
### MAKE 'logs' DIRECTORY BEFORE SBATCHING!
### GET constraint options: sinfo -N -a -h -o "%f" | tr ',' '\n' | sort -u