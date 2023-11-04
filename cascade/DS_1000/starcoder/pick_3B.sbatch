#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=13B_ds

module purge

MODEL=1
TEST_LINES=1
NUM_LOOPS=5

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cascade

export TRANSFORMERS_CACHE="/scratch/ax2047/huggingface_cache"
python pick_at_k.py --model=$MODEL --test_lines=$TEST_LINES --num_loops=$NUM_LOOPS --pass_at=$SLURM_ARRAY_TASK_ID
