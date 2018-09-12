#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=maxdiff
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

echo Hello

cd /om/user/sanjanas/minimal-images
singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/sanjanas/belledon-tensorflow-keras-master-latest.simg \
python find_maxdiff_tester.py
