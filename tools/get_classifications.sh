#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=statistics
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=use-everything

declare -a MODELS=("inception" "resnet")

cd /om/user/sanjanas/minimal-images
for MODEL in "${MODELS[@]}"; do
    echo $MODEL
    singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/sanjanas/belledon-tensorflow-keras-master-latest.simg \
    python minimal-image-statistics.py $MODEL
done 
