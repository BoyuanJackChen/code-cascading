#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --time=00:20:00
#SBATCH --mem=100GB
#SBATCH --job-name=install_all_models

module purge

singularity exec --nv \
            --overlay /vast/bc3194/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; python -u install_all_models.py"
