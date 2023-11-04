#!/bin/bash

#SBATCH -q nvidia-xxl
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=2B_cg_he

module purge

MODEL=1
TEST_LINES=1

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cascade

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
python -u pick_at_k.py --model=1 --test_lines=1 --num_loops=100 --pass_at=${SLURM_ARRAY_TASK_ID}
