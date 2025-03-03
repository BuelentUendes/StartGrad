#!/bin/bash

# Define the array of iteration values
iterations=(5 10 50 75 100 150 200 300)  # This is for the longer end of the experimental scope

# shellcheck disable=SC2068
# shellcheck disable=SC1073
# shellcheck disable=SC1061

# Loop over each iteration value
for iteration in "${iterations[@]}"
do
    # Run the Python script with the specified parameters
    python3 ./../main_vision.py --method pixelRDE,pixelRDE_saliency,pixelRDE_uniform,waveletX,waveletX_saliency,waveletX_uniform,shearletX,shearletX_saliency,shearletX_uniform,IG,GradCAM --iterations "$iteration" --number_samples 100 --gpu 1 --get_faithfulness
done