# Main script for time-series mask-based explanation methods

import os
import argparse
import json
import pickle
import multiprocessing

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from src.time_series.XAI_classifier import StateClassifier, PerturbationNetwork
from src.time_series.timeseries_mask_explainer import ExtremaMaskExplainer
from utils.general.helper_path import MODELS_PATH, CONFIG_PATH, RESULTS_PATH, FIGURES_PATH, TIME_SERIES_PATH
from utils.general.helper import create_directory, seed_everything, load_yaml_config_file
from utils.time_series.helper_state_dataset_fit import create_state_dataset, standardize_input
from utils.time_series.helper_switch_dataset import create_switch_dataset
from utils.time_series.helper_train_timeseries import TimeSeriesDataset, train_model
from utils.time_series.helper_timeseries_metrics import get_aup_and_aur, get_mask_information, get_mask_entropy
from utils.time_series.helper_plotting import get_mask_histogram


FEATURE_DIM = 3
HIDDEN_DIM = 200


def load_pretrained_model(args):
    model = StateClassifier(rnn_type=args.model_type, feature_size=FEATURE_DIM, hidden_size=HIDDEN_DIM)
    save_name = args.save_name if args.save_name else args.model_type
    pretrained_file_name = os.path.join(MODELS_PATH, str(args.seed), save_name + ".ckpt")

    try:
        model.load_state_dict(torch.load(pretrained_file_name))
        print("Loaded the model correctly")
    except FileNotFoundError:
        print(f"We could not find the file {pretrained_file_name} for the pretrained model weights."
              f"\nPlease make sure to train the model first and store the weights accordingly!")
        raise
    return model


def save_results(results, folder_path, save_name, format="json"):
    if format == "json":
        with open(os.path.join(folder_path, save_name) + ".json", "w") as f:
            json.dump(results, f)
    elif format == "pt":
        torch.save(results, os.path.join(folder_path, save_name) + ".pt")
    else:
        raise ValueError(f"Unsupported format {format}. Options. 'json' and 'pt'.")


def load_performance_metrics_history(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name + ".json")

    with open(file_path, "r") as f:
        mask_history = json.load(f)

    return mask_history


def load_history_performance_metrics(base_path, file_name):
    # Get all files in the base_path
    subfolders = os.listdir(base_path)
    # Select only directories, this is a list of strings (seed numbers)
    fold_folders = [item for item in subfolders if os.path.isdir(os.path.join(base_path, item))]
    # Sort the seed folders numerically
    fold_folders = sorted(fold_folders, key=lambda x: int(x))

    history_performance_metrics_across_folds = [
        load_performance_metrics_history(os.path.join(base_path, seed), file_name) for seed in fold_folders
    ]

    return history_performance_metrics_across_folds


def get_config_file_explainers(args):
    folder_path = os.path.join(CONFIG_PATH, "time_series", args.mode)
    hparams_baseline_explainer = load_yaml_config_file(os.path.join(folder_path, "hparams_extrema.yaml"))
    hparams_gradient_explainer = load_yaml_config_file(os.path.join(folder_path, "hparams_extrema_gradient.yaml"))

    return hparams_baseline_explainer, hparams_gradient_explainer


def get_performance_metrics(attr, true_saliency, iterations, verbose=True, init_method="gradient"):

    results = {
        "Number_iterations": iterations,
        "AUP": get_aup_and_aur(attr, true_saliency)[0],
        "AUR": get_aup_and_aur(attr, true_saliency)[1],
        "Information": get_mask_information(attr, true_saliency),
        "Entropy": get_mask_entropy(attr, true_saliency),
    }
    if verbose:
        print(f"{init_method}: {results}")

    return results


def get_mask_similarity(mask_a, mask_b):
    # Masks are of dimension batch, sequence_len, dim
    mask_coefficients = mask_a.shape[1] * mask_a.shape[2]
    difference = (mask_a.detach() - mask_b.detach()).abs()
    # We need to do it across the batch dimension, and divide with mask coefficients to be in range [0, 1]
    # A value close of 0 indicates similar
    mask_similarity = difference.sum(axis=(1, 2)) / mask_coefficients

    # We want to have 1 indicating very similar and 0 not, so 1-mask_similarity will give us this
    mask_similarities = 1 - mask_similarity
    mean_mask_similarity = torch.mean(mask_similarities).item()

    return mask_similarities, round(mean_mask_similarity, 4)


def get_performance_metrics_per_iteration(mask_history, true_saliency, iteration_steps=25):
    # We added the 0th mask as well, so we have iterations + 1 in the mask_history
    assert len(mask_history) >= iteration_steps, "iteration steps are greater than the length of the data!"
    assert (len(mask_history) - 1) % iteration_steps == 0, "iteration steps should be a multiple of the overall data len"

    subset_masks = mask_history[::iteration_steps]

    # Check if it worked
    assert len(subset_masks) == (((len(mask_history) - 1) / iteration_steps) + 1)

    results = {
        "AUP": [get_aup_and_aur(attr, true_saliency)[0] for attr in subset_masks],
        "AUR": [get_aup_and_aur(attr, true_saliency)[1] for attr in subset_masks],
        "Information": [get_mask_information(attr, true_saliency) for attr in subset_masks],
        "Entropy": [get_mask_entropy(attr, true_saliency) for attr in subset_masks],
    }

    return results


def get_mean_std_performance_metrics(performance_metrics_history):
    """
    Returns a dictionary, where for each metric the first is the mean, the second the std deviation
    :param performance_metrics_history: list of performance metrics histories across different folds
    :return: dictionary with mean and std deviation for each metric across folds
    """

    aup_results = np.array([fold_performance["AUP"] for fold_performance in performance_metrics_history])
    aur_results = np.array([fold_performance["AUR"] for fold_performance in performance_metrics_history])
    information_results = np.array([seed_performance["Information"] for seed_performance in
                                    performance_metrics_history])
    entropy_results = np.array([seed_performance["Entropy"] for seed_performance in
                                performance_metrics_history])

    results = {
        "AUP": (aup_results.mean(axis=0), aup_results.std(axis=0, ddof=0)), # baseline papers take also ddof=0
        "AUR": (aur_results.mean(axis=0), aur_results.std(axis=0, ddof=0)),
        "Information": (information_results.mean(axis=0), information_results.std(axis=0, ddof=0)),
        "Entropy": (entropy_results.mean(axis=0), entropy_results.std(axis=0, ddof=0)),
    }

    return results


def plot_performance_metrics_over_iterations(model_results,
                                             iteration_steps,
                                             save_path,
                                             save_name="performance_metrics_over_time",
                                             average_plot=False,
                                             labels=("uniform initialization (baseline)",
                                                     "gradient initialization (ours)",
                                                     "StartGrad (ours)",
                                                     ),
                                             ):

    colors = {
        'black': '#000000',
        'All-ones': '#E69F00', #Orange
        # 'sky_blue': '#56B4E9', #Sky blue
        'StartGrad (ours)': '#56B4E9',  # Sky blue
        'Uniform (baseline)': '#009E73', #Green
        'yellow': '#F0E442',
        'blue': '#0072B2', # Blue
        'brown': '#D55E00',
        'magenta': '#CC79A7'
    }

    linestyles={
        "StartGrad (ours)": '-',
        "All-ones": '--',
        "Uniform (baseline)": ':'
    }

    metrics = list(model_results[0].keys())
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    # Decrease the size of x-ticks and y-ticks
    axs = axs.flatten()  # Flatten the 2D array of axes to iterate easily

    # Collecting all line objects for the legend
    lines = []
    legend_labels = []

    for i, metric in enumerate(metrics):
        if i == 0:
            line_list = []

        for label_idx, result in enumerate(model_results):
            result_model = result[metric]
            if label_idx == 2:
                color = colors['StartGrad (ours)']
                style = linestyles["StartGrad (ours)"]
            elif label_idx == 0:
                color = colors['Uniform (baseline)']
                style = linestyles["Uniform (baseline)"]
            else:
                color = colors["All-ones"]
                style = linestyles["All-ones"]

            # Label accordingly to visualize that it was scaled
            if metric == "AUP":
                metric_name = r"AUP $\mathbf{\uparrow}$"

            elif metric == "AUR":
                metric_name = r"AUR $\mathbf{\uparrow}$"

            elif metric == "Information":
                metric_name = r"Information [$10^{4}$]$\mathbf{\uparrow}$"

            elif metric == "Entropy":
                metric_name = r"Entropy [$10^{3}$]$\mathbf{\downarrow}$"

            elif metric == "Accuracy":
                metric_name = r"Accuracy $\mathbf{\downarrow}$"

            elif metric == "CE":
                metric_name = r"Cross-Entropy $\mathbf{\uparrow}$"

            elif metric == "Comprehensiveness":
                metric_name = r"Comprehensiveness $\mathbf{\uparrow}$"

            elif metric == "Sufficiency":
                metric_name = r"Sufficiency $\mathbf{\downarrow}$"

            if average_plot:
                # results model is a tuple (mean, std)
                iterations = np.arange(0, len(result_model[0]) * iteration_steps, iteration_steps)

            else:
                iterations = np.arange(0, len(result_model) * iteration_steps, iteration_steps)

            if average_plot:
                # Then we plot the average and the uncertainty bands
                # results model is a tuple (mean, std)
                line, = axs[i].plot(iterations, result_model[0], label=labels[label_idx],
                                    color=color, linestyle=style)
                axs[i].fill_between(iterations, result_model[0] - result_model[1],
                                    result_model[0] + result_model[1], alpha=0.2, color=color)

            else:
                line, = axs[i].plot(iterations, result_model, label=labels[label_idx], color=color)

            line_list.append(line)

            axs[i].set_xlim(1, iterations[-1])
            axs[i].set_xticks(iterations)
            axs[i].set_xlabel("Number of iterations", fontsize=12)
            axs[i].set_ylabel(f"{metric_name}", fontsize=12)

        lines.extend(line_list)
        if i == 0:
            legend_labels.extend(labels)

    # Create a single legend for the entire figure
    fig.legend(lines, legend_labels, loc='lower center', ncols=len(labels), fontsize=12, facecolor="white")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_path, f"{save_name}.pdf"), dpi=300, format="pdf")
    plt.show()
    plt.close()


def combine_multiprocessing_results(results, target_sample_size):
    observations = torch.cat([res[0] for res in results])
    y = torch.cat([res[1] for res in results])
    important_features = torch.cat([res[2] for res in results])

    # Double-check that everything is correct
    assert observations.shape[0] == target_sample_size
    assert y.shape[0] == target_sample_size
    assert important_features.shape[0] == target_sample_size

    return observations, y, important_features


def load_or_create_dataset(args):
    target_path = os.path.join(TIME_SERIES_PATH, f"{args.dataset}", str(args.seed))
    create_directory(target_path)
    file_path = os.path.join(target_path, "dataset.pkl")
    if os.path.exists(file_path):
        print(f"We found the dataset at {file_path}. We will load it")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    else:
        num_processes = multiprocessing.cpu_count()
        print(f"We are using a total of {num_processes} cpus!")
        chunk_size = args.sample_size // num_processes
        remaining_samples = args.sample_size % num_processes

        pool = multiprocessing.Pool(num_processes)
        tasks = [(args.signal_length, chunk_size) for _ in range(num_processes)]

        if remaining_samples > 0:
            tasks.append((args.signal_length, remaining_samples))

        if args.dataset == "state":
            results = pool.starmap(create_state_dataset, tasks)
        elif args.dataset == "switch":
            results = pool.starmap(create_switch_dataset, tasks)

        pool.close()
        pool.join()

        observations, y, important_features = combine_multiprocessing_results(results, args.sample_size)

        with open(file_path, "wb") as f:
            pickle.dump((observations, y, important_features), f)
        return observations, y, important_features


def main(args):
    base_path_results = os.path.join(RESULTS_PATH, "time_series", args.dataset, "extremal",
                                     str(args.seed), args.model_type, args.mode)
    base_path_figures = os.path.join(FIGURES_PATH, "time_series", args.dataset, "extremal",
                                     str(args.seed), args.model_type, args.mode)
    observations, y, important_features = load_or_create_dataset(args)

    create_directory(base_path_results)
    create_directory(base_path_figures)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_splits = [k for k in kf.split(observations)]

    if args.get_mask_hist:
        mask_history_dict = {"gradient": []}

    for fold, (train_idx, test_idx) in enumerate(all_splits):
        print(f"Processing fold: {fold + 1}")
        folder_path_results = os.path.join(base_path_results, str(fold))
        create_directory(folder_path_results)

        if args.plot:
            folder_path_figures = os.path.join(base_path_figures, str(fold))
            create_directory(folder_path_figures)

        assert len(test_idx) == int((args.sample_size / args.n_folds))

        train_observations = observations[train_idx]
        y_train = y[train_idx]

        test_observations = observations[test_idx]
        y_test = y[test_idx]
        true_saliency = important_features[test_idx]

        if args.standardize:
            train_observations, train_mean, train_std = standardize_input(train_observations)
            test_observations, _, _ = standardize_input(test_observations)

        if args.model_type in ("GRU", "LSTM"):
            model = StateClassifier(FEATURE_DIM, args.hidden_size,
                                    rnn_type=args.model_type).to(DEVICE)

        else:
            raise ValueError(f"{args.model_type} not supported. Please use either 'GRU' or 'LSTM'")

        dataset_train = TimeSeriesDataset(train_observations, y_train)
        dataset_test = TimeSeriesDataset(test_observations, y_test)

        dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, drop_last=False)
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, drop_last=False)

        model = train_model(model, dataloader_train, dataloader_test, lr=1e-4, epochs=args.epochs,
                            verbose=args.verbose, device=DEVICE)

        # Original baseline paper sets also hidden size equal to feature dim,
        # see: https://github.com/josephenguehard/time_interpret/blob/main/experiments/hmm/main.py
        perturbation_network = PerturbationNetwork(feature_size=FEATURE_DIM, hidden_size=FEATURE_DIM,
                                                   signal_length=args.signal_length).to(DEVICE)

        # Extremal Mask explainers
        explainer_baseline = ExtremaMaskExplainer(model, perturbation_network, device=DEVICE)
        explainer_ones = ExtremaMaskExplainer(model, perturbation_network, device=DEVICE)
        explainer_startgrad = ExtremaMaskExplainer(model, perturbation_network, device=DEVICE)

        # Now we can get the explanations
        mask_baseline, mask_history_baseline = explainer_baseline.attribute(
            test_observations, iterations=args.iterations, verbose=args.verbose,
            mask_initialization_method="uniform", mode=args.mode, scaling=args.scaling)

        mask_baseline_ones, mask_history_baseline_ones = explainer_ones.attribute(
            test_observations, iterations=args.iterations, verbose=args.verbose,
            mask_initialization_method="ones", mode=args.mode, scaling=args.scaling)

        mask_gradient_quantile, mask_history_gradient_quantile = explainer_startgrad.attribute(
            test_observations, iterations=args.iterations, verbose=args.verbose, transformation="quantile_transformation",
            noisy_gradients=args.noisy_gradients, gradient_noise=args.noise,
            mask_initialization_method="gradient_based", mode=args.mode)

        if args.get_mask_hist:
            extremal_mask_explainer_gradient = ExtremaMaskExplainer(model, perturbation_network, device=DEVICE)
            mask_gradient, mask_history_gradient = extremal_mask_explainer_gradient.attribute(
                test_observations, iterations=0, verbose=args.verbose, transformation="identity",
                mask_initialization_method="gradient_based", mode=args.mode, scaling=False)
            mask_history_dict["gradient"].append(mask_gradient)

        results_baseline_over_iterations = get_performance_metrics_per_iteration(mask_history_baseline, true_saliency,
                                                                                 iteration_steps=args.iteration_steps)

        results_baseline_over_iterations_ones = get_performance_metrics_per_iteration(
            mask_history_baseline_ones, true_saliency, iteration_steps=args.iteration_steps)

        results_gradient_over_iterations_quantile = get_performance_metrics_per_iteration(mask_history_gradient_quantile,
                                                                                          true_saliency,
                                                                                 iteration_steps=args.iteration_steps)

        results_baseline = get_performance_metrics(mask_baseline, true_saliency, args.iterations, verbose=True,
                                                   init_method="uniform")
        results_baseline_ones = get_performance_metrics(mask_baseline_ones, true_saliency, args.iterations,
                                                        verbose=True, init_method="ones")
        results_gradient_quantile = get_performance_metrics(mask_gradient_quantile, true_saliency,
                                                            args.iterations, verbose=True, init_method="gradient")

        save_results(results_baseline, folder_path_results, "baseline_uniform", format="json")
        save_results(results_baseline_ones, folder_path_results, "baseline_ones", format="json")
        save_results(results_gradient_quantile, folder_path_results, "gradient_init_quantile", format="json")

        save_results(results_baseline_over_iterations, folder_path_results, "baseline_init_iterations", format="json")
        save_results(results_baseline_over_iterations_ones, folder_path_results,
                     "baseline_init_iterations_ones", format="json")
        save_results(results_gradient_over_iterations_quantile, folder_path_results,
                     "gradient_init_iterations_quantile", format="json")

        save_results(mask_history_baseline, folder_path_results, save_name="baseline_init_mask_history", format="pt")
        save_results(mask_history_baseline_ones, folder_path_results,
                     save_name="baseline_init_mask_history_ones", format="pt")
        save_results(mask_history_gradient_quantile, folder_path_results,
                     save_name=f"gradient_init_mask_history_quantile", format="pt")

        labels = (
            "uniform initialization (baseline)",
            "all-ones initialization",
            "StartGrad (ours)",
        )

        model_results = [
            results_baseline_over_iterations,
            results_baseline_over_iterations_ones,
            results_gradient_over_iterations_quantile,
        ]

        if args.plot:
            plot_performance_metrics_over_iterations(model_results,
                                                     args.iteration_steps,
                                                     save_name=args.save_name,
                                                     save_path=folder_path_figures,
                                                     labels=labels,
                                                     )

    if args.get_mask_hist:
        get_mask_histogram(mask_history_dict, base_path_figures)

    if args.plot_average:
        create_directory(base_path_figures)

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

        model_results = [
            performance_baseline_mean_std,
            performance_baseline_mean_std_ones,
            performance_gradient_quantile_mean_std,
        ]

        labels = (
            "uniform initialization (baseline)",
            "all-ones initialization",
            "StartGrad (ours)",
        )

        plot_performance_metrics_over_iterations(model_results,
                                                 args.iteration_steps,
                                                 save_path=base_path_figures,
                                                 save_name=f"{args.model_type}_{args.dataset}_{args.mode}_performance_metrics_over_time_avg",
                                                 average_plot=True,
                                                 labels=labels,
                                                 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--iteration_steps", type=int, default=5,
                        help="Used for plotting. How many iteration steps to use for plotting.")
    parser.add_argument("--model_type", type=str, default="GRU", choices=("GRU, LSTM"))
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="state", choices=("state, switch"))
    parser.add_argument("--signal_length", type=int, default=200, help="Length of the time_series")
    parser.add_argument("--sample_size", type=int, default=1_000, help="Number of timeseries to generate")
    parser.add_argument("--n_folds", type=int, help="Number of folds.", default=5)
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize input features. "
                             "IMPORTANT: Standardization leads to worse performance of the ExtremalMask approach!")
    parser.add_argument("--save_name", type=str, help="Name under which to save the output (optional)")
    parser.add_argument("--mode", type=str, default="preservation_game", choices=("preservation_game, deletion_game"),
                        help="Which optimization type to run the experiments with.")
    parser.add_argument("--updated_perturbation_loss", action="store_true",
                        help="Boolean. If we should use improved perturbation loss term as suggested in TMLR 2024. "
                             "Important: Only used when 'mode' is set to deletion game.")
    parser.add_argument("--plot", action="store_true", help="plots performance for this single run")
    parser.add_argument("--get_mask_hist", action="store_true", help="Get the histogram of mask initializations.")
    parser.add_argument("--noisy_gradients", action="store_true", help="Boolean. If noisy gradients should be used")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--scaling", action="store_true",
                        help="Boolean. If true, we min-max scale the gradient signal."
                             "Only applied for non-gradient-based mask initialization schemes")
    parser.add_argument("--plot_average", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Verbose training output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gpu', help='Int. Which gpu to use (if available).', type=int, default=1)

    args = parser.parse_args()
    seed_everything(args.seed)
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    print(f"GPU {DEVICE} is used")
    main(args)





