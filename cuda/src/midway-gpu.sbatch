#!/bin/bash
#SBATCH --job-name=run_gpu
#SBATCH --output=run_gpu.log
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/8.0

./my_awesome_program