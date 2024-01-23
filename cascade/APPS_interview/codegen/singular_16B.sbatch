#!/bin/bash

#SBATCH -q ebrainccs
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=71:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=16B_cg_apps

module purge

MODEL=3
NUM_LOOPS=10

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate wizard

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
export HF_HOME="/scratch/bc3194/huggingface_cache"
python -u singular.py --model=$MODEL --num_loops=$NUM_LOOPS --pass_at=${SLURM_ARRAY_TASK_ID}
