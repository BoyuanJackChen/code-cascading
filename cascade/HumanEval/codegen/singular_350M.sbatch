#!/bin/bash

#SBATCH -q nvidia-xxl
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:59
#SBATCH --mem=50GB
#SBATCH --job-name=350M_sc_he

module purge

MODEL=0
NUM_LOOPS=10

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate wizard

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
python -u singular.py --model=$MODEL --num_loops=$NUM_LOOPS --pass_at=$SLURM_ARRAY_TASK_ID
