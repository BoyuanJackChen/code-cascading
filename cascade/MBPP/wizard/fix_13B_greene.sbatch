#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=13B_fix_mbpp

module purge

MODEL=3
TEST_LINES=1
NUM_LOOPS=10

singularity exec --nv \
            --overlay /vast/bc3194/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; export TRANSFORMERS_CACHE='/vast/bc3194/huggingface_cache'; python -u fix_bad.py --model=$MODEL --test_lines=$TEST_LINES --num_loops=$NUM_LOOPS --pass_at="${SLURM_ARRAY_TASK_ID}
            