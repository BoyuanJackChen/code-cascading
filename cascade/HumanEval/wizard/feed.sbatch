#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=feed

module purge

MODEL=2
TEST_LINES=1

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cascade

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
python -u feed.py --model=3 --test_lines=1 --num_loops=1 --pass_at=1
