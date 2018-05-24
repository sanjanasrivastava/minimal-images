#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=maxdiff
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=use-everything

cd /om/user/sanjanas/minimal-images
python find_maxdiff_tester.py
