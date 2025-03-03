#!/bin/bash

# Bash script for running several experiments
pretrained_models=("resnet18")
NUMBER_SAMPLES=100 #100 samples only as we ran over a lot seeds, but it is not to be expected that this will change the results
ITERATIONS=300
seeds=(1 3 5 7)
# shellcheck disable=SC2068
# shellcheck disable=SC1073
# shellcheck disable=SC1061
for model in "${pretrained_models[@]}"
do
  for i in "${seeds[@]}"
  do
    python3 ./../main_vision.py --iterations $ITERATIONS --folder ImageNet --input all --number_samples $NUMBER_SAMPLES --method shearletX,shearletX_saliency,shearletX_uniform --wandb_logging --seed $i --pretrained_model $model --gpu 0
  done
done

