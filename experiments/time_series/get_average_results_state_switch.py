# Script to run and results across different random seeds

import os
import argparse
import json

import torch
import numpy as np

from utils.general.helper_path import RESULTS_PATH
from utils.general.helper import seed_everything
from main_time_series import load_history_performance_metrics, get_mean_std_performance_metrics


def get_metric_results(data, metric="AUP", idx=2):
    # This is 0th, 25th iteration etc,
    return np.round(data[metric][0][idx], 3), np.round(data[metric][1][idx], 3)  # every 5th result


def main(args):

    base_path_results = os.path.join(RESULTS_PATH, "time_series", args.dataset, str(args.seed), args.model_type,
                                     args.mode)
    save_path = os.path.join(RESULTS_PATH, "time_series", args.dataset)

    # First get  the AUP scores
    # 50th iteration, therefore 0, 25, 50
    # Iteration steps is important: we had for seed 1,3,5,7,10 iteration steps of 50 so 500/50 = 10

    performance_dict = {
        "AUP": {"Uniform": [], "Ones": [], "StartGrad": []},
        "AUR": {"Uniform": [], "Ones": [], "StartGrad": []},
        "Information": {"Uniform": [], "Ones": [], "StartGrad": []},
        "Entropy": {"Uniform": [], "Ones": [], "StartGrad": []},
    }

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        base_path_results = os.path.join(RESULTS_PATH, "time_series", args.dataset, str(seed), args.model_type,
                                         args.mode)

        performance_metrics_history_baseline = load_history_performance_metrics(base_path_results,
                                                                                "baseline_init_iterations")
        performance_metrics_history_baseline_ones = load_history_performance_metrics(base_path_results,
                                                                                     "baseline_init_iterations_ones")
        performance_metrics_history_gradient_quantile = load_history_performance_metrics(base_path_results,
                                                                                         "gradient_init_iterations_quantile")

        performance_baseline_mean_std = get_mean_std_performance_metrics(performance_metrics_history_baseline)
        performance_baseline_mean_std_ones = get_mean_std_performance_metrics(performance_metrics_history_baseline_ones)
        performance_gradient_quantile_mean_std = get_mean_std_performance_metrics(
            performance_metrics_history_gradient_quantile)

        for metric in ["AUP", "AUR", "Information", "Entropy"]:
            performance_dict[metric]["Uniform"].append([
                get_metric_results(performance_baseline_mean_std, metric=metric, idx=idx_nr)[0] for idx_nr in [1,2,6,-1]
            ])
            performance_dict[metric]["Ones"].append([
                get_metric_results(performance_baseline_mean_std_ones, metric=metric, idx=idx_nr)[0] for idx_nr in [1,2,6,-1]
            ])
            performance_dict[metric]["StartGrad"].append([
                get_metric_results(performance_gradient_quantile_mean_std, metric=metric, idx=idx_nr)[0]
                for idx_nr in [1, 2, 6, -1]
            ])

    # # Now we get the averages across the random seeds:
    performance_avg_seed = {
        "Uniform": {metric: [] for metric in ["AUP", "AUR", "Information", "Entropy"]},
        "Ones": {metric: [] for metric in ["AUP", "AUR", "Information", "Entropy"]},
        "StartGrad": {metric: [] for metric in ["AUP", "AUR", "Information", "Entropy"]}
    }

    for metric in ["AUP", "AUR", "Information", "Entropy"]:
        for initialization_method, values in performance_dict[metric].items():
            # Important: metric values for information and entropy are only 10^4 and 10^3 for entropy
            # So we need to still divide these!
            performance_avg_seed[initialization_method][metric] = (
                np.round(np.asarray(values).mean(axis=0), 3), np.round(np.asarray(values).std(axis=0), 3)
            )

    data_serializable = {
        key: {
            sub_key: (value[0].tolist(), value[1].tolist()) for sub_key, value in sub_dict.items()
        } for key, sub_dict in performance_avg_seed.items()
    }

    with open(os.path.join(save_path,
                           f'avg_seed_results_time_series.json_{args.dataset}_{args.mode}'), 'w') as json_file:
        json.dump(data_serializable, json_file, indent=4)
        print("Data is saved")

    print(performance_avg_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--iteration_steps", type=int, default=50,
                        help="Used for plotting. How many iteration steps to use for plotting.")
    parser.add_argument("--model_type", type=str, default="GRU", choices=("GRU, LSTM, TCN"))
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="state", choices=("state, switch"))
    parser.add_argument("--signal_length", type=int, default=200, help="Length of the time_series")
    parser.add_argument("--sample_size", type=int, default=1_000, help="Number of timeseries to generate")
    parser.add_argument("--n_folds", type=int, help="Number of folds.", default=5)
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize input features. "
                             "IMPORTANT: Standardization leads to worse performance of the ExtremalMask approach!")
    parser.add_argument("--save_name", type=str, help="Name under which to save the output (optional)")
    parser.add_argument("--mode", type=str, default="deletion_game", choices=("preservation_game, deletion_game"),
                        help="Which optimization type to run the experiments with.")
    parser.add_argument("--mask_init", type=str, default="gradient_based",
                        choices=("gradient_based, gradient_x_based, smoothgrad, uniform, ones"),
                        help="Which mask init for the gradient-method to use.")
    parser.add_argument("--updated_perturbation_loss", action="store_true",
                        help="Boolean. If we should use improved perturbation loss term as suggested in TMLR 2024. "
                             "Important: Only important for deletion game.")
    parser.add_argument("--plot", action="store_true", help="plots performance for this single run")
    parser.add_argument("--get_mask_hist", action="store_true", help="Get the histogram of mask initializations.")
    parser.add_argument("--noisy_gradients", action="store_true", help="Boolean. If noisy gradients should be used")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--scaling", action="store_true",
                        help="Boolean. If true, we min-max scale the gradient signal."
                             "Only applied for non-gradient-based mask initialization schemes")
    parser.add_argument("--plot_average", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Verbose training output")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument('--gpu', help='Int. Which gpu to use (if available).', type=int, default=1)
    parser.add_argument("--metric", default="AUR")

    args = parser.parse_args()
    seed_everything(args.seed)
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    main(args)
