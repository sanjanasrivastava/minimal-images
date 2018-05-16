#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=statistics
#SBATCH --mem=10GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 0:30:00
#SBATCH --workdir=./log/
#SBATCH --qos=use-everything

declare -a CROP_METRICS=("0.2" "0.4" "0.6" "0.8")
# declare -a CROP_METRICS=("0.8")
declare -a MODELS=("vgg16" "resnet" "inception")
# declare -a MODELS=("inception")
declare -a IMAGE_SCALES=("1")
declare -a STRICTNESSES=("loose" "strict")
# declare -a STRICTNESSES=("loose")
declare -a AXES=("shift" "scale")
# declare -a AXES=("scale")

cd /om/user/sanjanas/minimal-images
for CROP_METRIC in "${CROP_METRICS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for IMAGE_SCALE in "${IMAGE_SCALES[@]}"; do
            for STRICTNESS in "${STRICTNESSES[@]}"; do
                for AXIS in "${AXES[@]}"; do
                     echo "$CROP_METRIC $MODEL $IMAGE_SCALE $STRICTNESS $AXIS"
                     singularity exec -B /om:/om -B /cbcl:/cbcl --nv /om/user/sanjanas/belledon-tensorflow-keras-master-latest.simg \
                     python minimal-image-statistics.py $CROP_METRIC $MODEL $IMAGE_SCALE $STRICTNESS $AXIS
                done
            done
        done
    done
done
