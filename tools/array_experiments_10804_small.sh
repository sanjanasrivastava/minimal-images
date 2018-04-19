#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0-500
#SBATCH --job-name=minimal
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=use-everything



SCALE=0.4
NETWORK=resnet

echo "$NETWORK"
echo "SCALE"

if [ ! -f /om/user/xboix/share/minimal-images/confidence/$SCALE/$NETWORK/1.0/${SLURM_ARRAY_TASK_ID}_small.npy ]; then
    /om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'
    cd /om/user/xboix/src/minimal-images
    singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/xboix/singularity/belledon-tensorflow-keras-master-latest.simg \
    python /om/user/xboix/src/minimal-images/confidence_maps_parallel_small.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_ID} $SCALE $NETWORK 1
fi