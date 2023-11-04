#!/bin/bash

#SBATCH -q ebrainccs
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=2B_1l_cg

module purge

MODEL=1
TEST_LINES=1

source ~/.bashrc
conda activate myenv1