#!/bin/bash

#SBATCH -q nvidia-xxl
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=2B_mbpp

module purge

MODEL=1
TEST_LINES=1
NUM_LOOPS=10

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cascade

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
python -u pick_at_k.py --model=$MODEL --test_lines=$TEST_LINES --num_loops=$NUM_LOOPS --pass_at=$SLURM_ARRAY_TASK_ID
