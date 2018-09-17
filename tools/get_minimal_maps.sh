#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=statistics
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 0:30:00
#SBATCH --workdir=./log/
#SBATCH --qos=use-everything

echo Hello

cd /om/user/sanjanas/minimal-images

singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/sanjanas/belledon-tensorflow-keras-master-latest.simg \
python minimal-image-statistics_small.py 0.4 resnet 1.0 loose scale