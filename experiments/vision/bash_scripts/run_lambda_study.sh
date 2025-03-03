#!/bin/bash

# Bash script for running several experiments
pretrained_models=("resnet18")
NUMBER_SAMPLES=500
ITERATIONS=300

# shellcheck disable=SC2068
# shellcheck disable=SC1073
# shellcheck disable=SC1061

for model in ${pretrained_models[@]}; do
  for l1 in 0.1 1.0 10.0; do
    for l2 in 0.1 1.0 10.0; do
      python3 ./../main_vision.py \
        --iterations $ITERATIONS \
        --number_samples $NUMBER_SAMPLES \
        --gpu 1 \
        --lambda_l1 $l1 \
        --lambda_l2 $l2 \
        --method waveletX,waveletX_uniform,waveletX_saliency \
        --wandb_logging \
        --pretrained_model "$model"
    done
  done
done
