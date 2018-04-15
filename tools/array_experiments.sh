#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0-10
#SBATCH --job-name=robustness
#SBATCH --mem=32GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 5:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'
cd /om/user/xboix/src/minimal-images
singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/xboix/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/xboix/src/minimal-images/confidence_map_parallel.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_ID} 0.2 inception 1


