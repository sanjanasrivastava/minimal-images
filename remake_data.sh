#!/bin/bash

# declare -a crop_metrics=("0.194" "0.2" "0.5" "224")
declare -a crop_metrics=("0.194")
# declare -a models=("vgg16" "resnet" "inception")
declare -a models=("inception")
declare -a image_scales=("0.5" "1.0")
# declare -a image_scales=("1.0")

for crop_metric in "${crop_metrics[@]}"
do 
    for model in "${models[@]}"
    do
        for image_scale in "${image_scales[@]}"
        do
            python confidence_maps_parallel.py 1 4 "$crop_metric" "$model" "$image_scale" 
            echo "\n\n\n\n\n"
        done
    done
done
