#!/bin/bash

# Bash script for running several experiments
xai_methods=("waveletX" "shearletX" "pixelRDE")
metrics=("cp_pixel" "cp_l1")
models=("resnet18" "vgg16" "swin_t" "vit_base")

# shellcheck disable=SC2068
# shellcheck disable=SC1073
# shellcheck disable=SC1073d
# shellcheck disable=SC1061
# shellcheck disable=SC1061
for model in ${models[@]}
do
  for method in ${xai_methods[@]}
  do
    for metric in ${metrics[@]}
    do
      python3 ./../create_visuals_bootstrapped.py --metric $metric --method $method --statistics interquartile_mean --model_type $model
    done
  done
done
