#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=34B_ds1000

module purge

MODEL=5
TEST_LINES=1
NUM_LOOPS=5

singularity exec --nv \
            --overlay /vast/bc3194/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; pip install tensorflow; export TRANSFORMERS_CACHE='/vast/bc3194/huggingface_cache'; python -u pick_at_k.py --model=$MODEL --test_lines=$TEST_LINES --num_loops=$NUM_LOOPS --pass_at="${SLURM_ARRAY_TASK_ID}
            