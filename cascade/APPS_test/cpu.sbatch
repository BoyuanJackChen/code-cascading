#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:59
#SBATCH --mem=50GB
#SBATCH --job-name=APPS_count

module purge

MODEL=1
TEST_LINES=1
NUM_LOOPS=10

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cascade

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
export HF_HOME="/scratch/bc3194/huggingface_cache"
python -u count_canonical_tokens.py