#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --time=00:10
#SBATCH --output=awesome_program_%j.txt

./my_awesome_program