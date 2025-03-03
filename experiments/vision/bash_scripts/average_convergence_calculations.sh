#!/bin/bash

# Bash script for running average convergence results
metrics=("cp_l1" "cp_pixel")
statistics=("interquartile_mean" "median")
xai_methods=("shearletX")
references=("all_ones" "uniform")

# shellcheck disable=SC2068
# shellcheck disable=SC1073d
# shellcheck disable=SC1061
# shellcheck disable=SC1061
for method in ${xai_methods[@]}
do
  for metric in ${metrics[@]}
  do
    for statistic in ${statistics[@]}
    do
      for reference in ${references[@]}
      do
        echo "Processing $method $metric $statistic $reference . Please wait!"
        python3 ./../convergence_calculations.py --metric $metric --method $method --bootstrap --statistics $statistic --reference $reference
      done
    done
  done
done

