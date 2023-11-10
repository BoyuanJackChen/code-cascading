#!/bin/bash

#SBATCH -q nvidia-xxl
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:59
#SBATCH --mem=50GB
#SBATCH --job-name=15B_sc_tapps

module purge

MODEL=2
NUM_LOOPS=10

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate wizard

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
export HF_HOME="/scratch/bc3194/huggingface_cache"
python -u testcase.py --model=$MODEL --num_loops=$NUM_LOOPS --pass_at=$SLURM_ARRAY_TASK_ID
