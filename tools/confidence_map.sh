#!/usr/bin/env bash
#SBATCH -n 2
#SBATCH --job-name=statistics
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

declare -a MODELS=("resnet")
echo Hello

cd /om/user/sanjanas/minimal-images
for MODEL in "${MODELS[@]}"; do
    echo $MODEL
    singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/sanjanas/belledon-tensorflow-keras-master-latest.simg \
    python confidence_maps_parallel.py 50002 50008 0.2 resnet 1
done
