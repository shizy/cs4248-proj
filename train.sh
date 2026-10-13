#!/bin/bash
#SBATCH --job-name=train_run
#SBATCH --output=output-%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=h100-96

source ./.venv/bin/activate

python train.py
