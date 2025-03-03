#!/bin/bash

# shellcheck disable=SC2068
models=("GRU")
seeds=(42) # Original seed as used in the paper and in previous studies
seeds=(1 3 5 7) # add 4 additional seeds to make it more comparable
modes=("preservation_game" "deletion_game")

for model in ${models[@]}
do
  for seed in ${seeds[@]}
  do
    for mode in ${modes[@]}
    do
      python3 main_time_series.py --model_type $model --mode $mode --dataset state --n_folds 5 --plot_average --updated_perturbation_loss --seed $seed --epochs 50 --iterations 500 --iteration_steps 50 --gpu 1
      python3 main_time_series.py --model_type $model --mode $mode --dataset switch --n_folds 5 --plot_average --updated_perturbation_loss --seed $seed --epochs 50 --iterations 500 --iteration_steps 50 --gpu 1
    done
  done
done
