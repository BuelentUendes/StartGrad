import argparse
import os

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from scipy import stats

from utils.general.helper_path import RESULTS_PATH, FIGURES_PATH
from utils.general.helper import create_directory


# Colorblind friendly version
# Website: https://scottplot.net/cookbook/4.1/colors/#category-10
colors = {
    'black':   '#000000',
    'orange':  '#E69F00',
    'sky_blue': '#56B4E9',
    'green':    '#009E73',
    'yellow':  '#F0E442',
    'blue':    '#0072B2',
    'brown':   '#D55E00',
    'magenta': '#CC79A7'
}


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def bootstrap_statistics(data: pd.Series, n_iterations: int = 500, statistics="median", q=0.5,
                         apply_percentile_filter=True, percentile=99) -> pd.DataFrame:
    """
    Perform bootstrapping to calculate the mean, median, and standard deviation.

    Args:
        data (pd.Series): The data to bootstrap.
        n_iterations (int): The number of bootstrap iterations.

    Returns:
        pd.DataFrame: DataFrame with mean, median, and std deviation from bootstrapped samples.
    """

    # Apply percentile filtering if requested
    if apply_percentile_filter and statistics == "mean":
        # Calculate the threshold value at the specified percentile
        upper_threshold_value = data.quantile(percentile / 100.0)
        lower_threshold_value = data.quantile(1 / 100.0)
        # Filter data to exclude values above the threshold
        data = data[data <= upper_threshold_value]
        data = data[data >= lower_threshold_value]

    if statistics == "interquartile_mean":
        upper_threshold_value = data.quantile(0.75)
        lower_threshold_value = data.quantile(0.25)
        data = data[data >= lower_threshold_value]
        data = data[data <= upper_threshold_value]

    data_values = data.values
    n = max(len(data_values), 500) # Either bootstrap with 500 or take simply the number of values
    n = len(data_values)
    indices = np.random.randint(0, len(data_values), (n_iterations, n))
    if statistics == "median":
        bootstrapped_performance = np.median(data_values[indices], axis=1)

    elif statistics == "quantile":
        bootstrapped_performance = np.quantile(data_values[indices], q=q, axis=1)

    elif statistics == "mean" or statistics == "interquartile_mean":
        bootstrapped_performance = np.mean(data_values[indices], axis=1)

    else:
        raise ValueError(f"{statistics} is not supported. Either 'median' or 'mean', "
                         f"'interquartile_mean', 'quantile'")

    return pd.DataFrame({
        f'{statistics}': bootstrapped_performance,
    })


def find_number_iterations_to_baseline(performance_series, baseline_threshold):
    """
    Identify the number of iterations to reach baseline performance
    :param performance_series:
    :param baseline_threshold:
    :return:
    """
    len_iterations = len(performance_series)
    if performance_series[-1] < baseline_threshold:
        number_iterations = len_iterations
    else:
        number_iterations = np.argmax(performance_series >= baseline_threshold)
    return number_iterations


def calculate_convergence_rates(df: pd.DataFrame, baseline_threshold):

    grouped = df.groupby("lineKey")
    convergence_rates = [
        find_number_iterations_to_baseline(group["lineVal"].values, baseline_threshold)
        for image, group in grouped
    ]

    return convergence_rates


def calculate_performance_metric_by_step(df: pd.DataFrame, n_iterations: int = 500,
                                         statistics="median", q=0.5,
                                         bootstrap=True,
                                         confidence_interval=True) -> pd.DataFrame:
    """
    Calculate bootstrapped mean, median, and std deviation for each step.

    Args:
        df (pd.DataFrame): DataFrame containing 'step', 'image_nr', and 'value'.
        n_iterations (int): Number of bootstrap samples.
        statistics: median or average to track the performance

    Returns:
        pd.DataFrame: DataFrame with aggregated bootstrapped statistics for each step.
    """
    bootstrapped_stats = {}
    grouped = df.groupby('step')

    for step, group in grouped:
        if bootstrap:
            stats = bootstrap_statistics(group['lineVal'], n_iterations, statistics, q)

            # Aggregate bootstrapped statistics (mean, median, std, median_std) for each step
            bootstrapped_stats[step] = {
                f'performance_{statistics}': stats[f'{statistics}'].mean(),
                'performance_std': stats[f'{statistics}'].std(),
            }

        else:
            data = group["lineVal"]
            if statistics == "average":
                # Check this, as we remove 99 and 1 of the outlier here per step, check with average performance calculation!
                upper_threshold_value = data.quantile(0.99)
                lower_threshold_value = data.quantile(0.01)
                data = data[data >= lower_threshold_value]
                data = data[data <= upper_threshold_value]
                performance_metric = data.mean()
                std = data.std()

            elif statistics == "median":
                performance_metric = data.median()
                std = data.std()

            elif statistics == "quantile":
                performance_metric = data.quantile(q=q)
                std = data.std()

            elif statistics == "interquartile_mean":
                upper_threshold_value = data.quantile(0.75)
                lower_threshold_value = data.quantile(0.25)
                data = data[data >= lower_threshold_value]
                data = data[data <= upper_threshold_value]
                performance_metric = data.mean()
                std = data.std()

            bootstrapped_stats[step] = {
                f'performance_{statistics}': performance_metric,
                'performance_std':  (std / np.sqrt(len(data))) if confidence_interval else std,
            }

    stats_df = pd.DataFrame(bootstrapped_stats).T
    stats_df.index.name = 'step'
    return stats_df


def calculate_performance_metric_by_image(df: pd.DataFrame):
    grouped = df.groupby('lineKey') # Group by image

    final_performance = [
        group["lineVal"].values[-1] for image, group in grouped
    ]

    max_performance = [
        np.max(group["lineVal"].values) for image, group in grouped
    ]

    # Exact time step when max performance occurs, needed for wilcoxon rank sign test
    occurence_max_performance = [
        np.argmax(group["lineVal"].values) for image, group in grouped
    ]

    return final_performance, max_performance, occurence_max_performance


def get_time_to_milestone(df_baseline, df_comparison, milestone=50):

    grouped = df_baseline.groupby('lineKey')  # Group by image
    grouped_comparison = df_comparison.groupby("lineKey")

    performance_at_milestone_baseline = [
        group["lineVal"].values[milestone] for image, group in grouped
    ]

    occurence_performance_milestone_comparison = []

    for idx, (_, group) in enumerate(grouped_comparison):
        baseline_performance_to_match = performance_at_milestone_baseline[idx]
        comparison_values = group["lineVal"].values
        milestone_reached = np.where(comparison_values >=baseline_performance_to_match)[0]

        if len(milestone_reached) > 0:
            occurence_performance_milestone_comparison.append(milestone_reached[0] + 1)  # 1-based index
        else:
            occurence_performance_milestone_comparison.append(None)  # No milestone reached in comparison

    return occurence_performance_milestone_comparison


def get_statistical_result(difference_values, alternative="greater", correction=True):
    _, p_signed_test_result = stats.wilcoxon(difference_values, alternative=alternative, correction=correction)
    return p_signed_test_result


def get_iterations_until_final_performance_baseline(df, baseline,
                                                    final_number_iterations=300,
                                                    baseline_idx=None):

    grouped_df = df.groupby("lineKey") # group by image
    number_iterations_needed = []
    # We calculate the speed-up as final_number_iterations-iterations_needed
    speed_up_per_image = []

    for idx, (image, group) in enumerate(grouped_df):
        performance_values = group["lineVal"].values
        comparison_performance = baseline[idx]
        mask = performance_values >= comparison_performance
        iterations = np.argmax(mask) + 1 if mask.any() else final_number_iterations # python indexing of 0

        # if iterations is 0, it means that no speed up occured
        number_iterations_needed.append(iterations)
        if baseline_idx is None:
            speed_up_per_image.append(final_number_iterations - iterations)
        else:
            # We calculate the speed-ups as ratio old/new, so a ratio above 1 is a speedup!
            if iterations < baseline_idx[idx]:
                speed_up_per_image.append(baseline_idx[idx] - (iterations-1))
            else:
                # if it did not reach the performance metrics up until final iterations, no speed up, we put
                speed_up_per_image.append(0)

    return number_iterations_needed, speed_up_per_image


def plot_retained_prob_sparsity(
        stats_df: pd.DataFrame,
        save_path: str,
        method="waveletX",
        statistics="average",
        retained_prob=False,
        show_plot=False,
):
    """
       Plots retained probability or sparsity.
    """

    colors = {
        'black': '#000000',
        'All-ones \n(baseline)': '#E69F00', #Orange
        'StartGrad (ours)': '#56B4E9',  # Sky blue
        'Uniform': '#009E73', #Green
        'yellow': '#F0E442',
        'blue': '#0072B2', # Blue
        'brown': '#D55E00',
        'magenta': '#CC79A7',
    }

    linestyles = {
        "StartGrad (ours)": '-',
        "All-ones \n(baseline)": '--',
        "Uniform": ':'
    }

    grouped = stats_df.groupby("Initialization")
    for name, group in grouped:
        # Apply color scheme
        label = name
        if retained_prob:
            performance_mean_series = group[f"retained probability {statistics}"].values
        else:
            performance_mean_series = group[f"retained information l1 {statistics}"].values
        steps = np.arange(0, len(performance_mean_series))
        plt.plot(steps, performance_mean_series, label=f'{label}', linestyle=linestyles[label],
                 color=colors[label])

        plt.xlabel(r'$\mathbf{Iterations}$', fontsize=16)
        if retained_prob:
            plt.ylabel("Retained probability", fontsize=16, fontweight="bold")
        else:
            plt.ylabel("Retained information L1", fontsize=16, fontweight="bold")

    plt.grid(False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    location = "lower right" if retained_prob else "upper right"
    plt.legend(fontsize=16, loc=location)
    if retained_prob:
        plt.savefig(os.path.join(save_path,
                                 f"Retained probability {method} {statistics}.pdf"), dpi=300, format="pdf")
    else:
        plt.savefig(os.path.join(save_path,
                                 f"Retained information l1 {method} {statistics}.pdf"), dpi=300, format="pdf")

    if show_plot:
        plt.show()
    plt.close()


def plot_performance_metric_by_step(stats_df: pd.DataFrame, save_path: str,
                                    model=None, metric: str = None, method: str = None,
                                    statistics="median", q=0.5,
                                    noisy=False,
                                    smoothgrad=False,
                                    min_max_comparison=False,
                                    shuffle=False,
                                    lambda_l1=False,
                                    lambda_l2=False,
                                    initialization=None,
                                    get_forgrad=None,
                                    show_plot=True,
                                    factor=1.0):
    """
    Plot bootstrapped statistics for each step.

    Args:
        stats_df (pd.DataFrame): DataFrame with aggregated statistics for each step.
        save_path (str): Path to save the plot.
        metric (str, optional): Metric label for the plot.
        method (str): which mask based explanation method
        statistics: either median or average
    """
    plt.figure(figsize=(8, 8))

    if statistics == "median":
        statistic_label = "Median"

    if statistics == "interquartile_mean":
        statistic_label = "IQM"

    if statistics == "mean":
        statistic_label = "Mean"

    colors = {
        'black': '#000000',
        'All-ones \n(baseline)': '#E69F00', #Orange
        'StartGrad (ours)': '#56B4E9',  # Sky blue
        'Uniform': '#009E73', #Green
        'yellow': '#F0E442',
        'blue': '#0072B2', # Blue
        'brown': '#D55E00',
        'magenta': '#CC79A7',
    }

    linestyles={
        "StartGrad (ours)": '-',
        "All-ones \n(baseline)": '--',
        "Uniform": ':'
    }

    if get_forgrad:

        colors = {
            'black': '#000000',
            'StartGrad + ForGrad': '#E69F00',  # Orange
            'StartGrad': '#56B4E9',  # Sky blue
            'All-ones': '#009E73',  # Green
            'yellow': '#F0E442',
            'blue': '#0072B2',  # Blue
            'brown': '#D55E00',
            'magenta': '#CC79A7',
        }

        linestyles = {
            "StartGrad": '-',
            "StartGrad + ForGrad": '--',
            "All-ones": ':'
        }

    if smoothgrad:
        colors = {
            'black': '#000000',
            'StartGrad + SmoothGrad': '#E69F00',  # Orange
            'StartGrad (baseline)': '#56B4E9',  # Sky blue
            'Uniform': '#009E73',  # Green
            'yellow': '#F0E442',
            'blue': '#0072B2',  # Blue
            'brown': '#D55E00',
            'magenta': '#CC79A7',
        }

        linestyles = {
            "StartGrad (baseline)": '-',
            "StartGrad + SmoothGrad": '--',
            "Uniform": ':'
        }

    if noisy:
        # Options for the noisy plotting version
        colors = {
            'black': '#000000',
            'StartGrad min_max transform': '#E69F00', #Orange
            'StartGrad (min_max_transform)': '#56B4E9',  # Sky blue
        }

        linestyles={
            "StartGrad (baseline)": '-',
            "StartGrad \n$\\sigma^{2} = 0.01$": '--',
            "StartGrad \n$\\sigma^{2} = 0.05$": ':',
            "StartGrad \n$\\sigma^{2} = 0.20$": "-.",
        }

    if shuffle:
        colors = {
            'black': '#000000',
            'StartGrad (uninformative gradients)': '#E69F00',  # Orange
            'StartGrad (baseline)': '#56B4E9',  # Sky blue
            'Uniform': '#009E73',  # Green
            'StartGrad (adversarial gradients)': 'black',
            'blue': '#0072B2',  # Blue
            'brown': '#D55E00',
            'magenta': '#CC79A7',
        }

        linestyles = {
            "StartGrad (baseline)": '-',
            "StartGrad (uninformative gradients)": '--',
            "Uniform": ':',
            "StartGrad (adversarial gradients)": '-',

        }

    if lambda_l2:
        if method in ["waveletX"]:
            colors = {
                r"$\lambda_{1}=1$, $\lambda_{2}=10$": '#56B4E9',
                r"$\lambda_{1}=1$, $\lambda_{2}=1$": '#E69F00',
                r"$\lambda_{1}=1$, $\lambda_{2}=0.1$": '#009E73',
            }

            linestyles = {
                r"$\lambda_{1}=1$, $\lambda_{2}=10$":'-',
                r"$\lambda_{1}=1$, $\lambda_{2}=1$": '--',
                r"$\lambda_{1}=1$, $\lambda_{2}=0.1$": ':',
            }

        else:
            colors = {
                r"$\lambda_{1}=1$, $\lambda_{2}=20$": '#56B4E9',
                r"$\lambda_{1}=1$, $\lambda_{2}=2$": '#E69F00',
                r"$\lambda_{1}=1$, $\lambda_{2}=0.2$": '#009E73',
            }

            linestyles = {
                r"$\lambda_{1}=1$, $\lambda_{2}=20$": '-',
                r"$\lambda_{1}=1$, $\lambda_{2}=2$": '--',
                r"$\lambda_{1}=1$, $\lambda_{2}=0.2$": ':',
            }

    if lambda_l1:
        if method in ["waveletX"]:

            colors = {
                r"$\lambda_{1}=1$, $\lambda_{2}=10$": '#56B4E9',
                r"$\lambda_{1}=0.1$, $\lambda_{2}=10$": '#E69F00',
                r"$\lambda_{1}=10$, $\lambda_{2}=10$": '#009E73',
            }

            linestyles = {
                r"$\lambda_{1}=1$, $\lambda_{2}=10$":'-',
                r"$\lambda_{1}=0.1$, $\lambda_{2}=10$": '--',
                r"$\lambda_{1}=10$, $\lambda_{2}=10$": ':',
            }

        elif method in ["shearletX"]:
            colors = {
                r"$\lambda_{1}=1$, $\lambda_{2}=2$": '#56B4E9',
                r"$\lambda_{1}=0.1$, $\lambda_{2}=2$": '#E69F00',
                r"$\lambda_{1}=10$, $\lambda_{2}=2$": '#009E73',
            }

            linestyles = {
                r"$\lambda_{1}=1$, $\lambda_{2}=2$":'-',
                r"$\lambda_{1}=0.1$, $\lambda_{2}=2$": '--',
                r"$\lambda_{1}=10$, $\lambda_{2}=2$": ':',
            }

        else:
            colors = {
                r"$\lambda_{1}=1$": '#56B4E9',
                r"$\lambda_{1}=0.1$": '#E69F00',
                r"$\lambda_{1}=10$": '#009E73',
            }

            linestyles = {
                r"$\lambda_{1}=1$":'-',
                r"$\lambda_{1}=0.1$": '--',
                r"$\lambda_{1}=10$": ':',
            }

    if min_max_comparison:
        colors = {
            'black': '#000000',
            'StartGrad (min-max scaling)': '#E69F00',  # Orange
            'StartGrad (baseline)': '#56B4E9',  # Sky blue
            'Uniform': '#009E73',  # Green
            'yellow': '#F0E442',
            'blue': '#0072B2',  # Blue
            'brown': '#D55E00',
            'magenta': '#CC79A7',
        }

        linestyles = {
            "StartGrad (baseline)": '-',
            "StartGrad (min-max scaling)": '--',
            "Uniform": ':'
        }

    grouped = stats_df.groupby("Initialization")
    for name, group in grouped:
        #Apply color scheme
        label = name
        performance_mean_series = group[f"performance_{statistics}"].values
        performance_std_series = group["performance_std"].values
        steps = np.arange(0, len(performance_mean_series))

        lower_bound = performance_mean_series - factor * performance_std_series
        upper_bound = performance_mean_series + factor * performance_std_series

        plt.plot(steps, performance_mean_series, label=f'{label}', linestyle=linestyles[label], color=colors[label])
        plt.fill_between(steps, lower_bound, upper_bound, color=colors[label], alpha=0.2)

        plt.xlabel(r'$\mathbf{Iterations}$', fontsize=16)

        if metric == "cp_pixel":
            metric_name = r"CP-Pixel"

        elif metric == "cp_l1":
            metric_name = r"CP-L1"

        elif metric == "cp_entropy" or metric == "cp_entropy_no_exp":
            metric_name = r"CP-Entropy"

        plt.ylabel(f"$\\mathbf{{{statistic_label}}} \\ \\mathbf{{{ metric_name}}}$", fontsize=16)

    plt.grid(False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    location = "lower right" if not shuffle else "upper left"
    plt.legend(fontsize=16, loc=location)
    if statistics == "quantile":
        if noisy:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} noisy.pdf"), dpi=300, format="pdf")

        elif smoothgrad:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} smoothgrad.pdf"), dpi=300,
                        format="pdf")
        elif shuffle:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} shuffle.pdf"), dpi=300,
                        format="pdf")

        else:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q}.pdf"), dpi=300, format="pdf")
    else:
        if noisy:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} noisy.pdf"), dpi=300, format="pdf")

        elif smoothgrad:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} smoothgrad.pdf"), dpi=300, format="pdf")

        elif shuffle:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} shuffle.pdf"), dpi=300,
                        format="pdf")

        elif lambda_l1:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} {initialization} lambda_l1.pdf"),
                        dpi=300,
                        format="pdf")

        elif lambda_l2:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} {q} {initialization} lambda_l2.pdf"),
                        dpi=300,
                        format="pdf")

        elif min_max_comparison:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics} min_max_comparison.pdf"),
                        dpi=300, format="pdf")

        else:
            plt.savefig(os.path.join(save_path,
                                     f"Comparison {model} {method} {metric} {statistics}.pdf"), dpi=400, format="pdf")

    if show_plot:
        plt.show()

    plt.close()


def calculate_time_to_metric(df_one, q=0.25, max_score=False):
    """
    Function to calculate how fast a specific metric is achieved
    """
    grouped = df_one.groupby('lineKey') # Group by image

    if max_score:
        occurence_performance = [
            np.argmax(group["lineVal"].values) for idx, (_, group) in enumerate(grouped)
        ]

    else:
        threshold = [
            np.quantile(group["lineVal"].values, q=q) for _, group in grouped
        ]
        # Exact time step when max performance occurs, needed for wilcoxon rank sign test
        occurence_performance = [
            np.argmax(group["lineVal"].values > threshold[idx]) for idx, (_, group) in enumerate(grouped)
        ]

    return occurence_performance


def calculate_time_step_difference(time_baseline, time_alternative):   # Speed up in reaching its maximum performance
    # Positive value means we reach it much faster, negative later!
    difference_peak_timestep = [
        time_a - time_b for (time_a, time_b) in zip(time_baseline, time_alternative)
    ]
    return difference_peak_timestep


def find_convergence_point(series, abs_threshold=None, rel_threshold=None):
    for i in range(1, len(series)):
        # Check absolute or relative threshold condition
        if abs_threshold is not None:
            if all(abs(series[j] - series[j - 1]) < abs_threshold for j in range(i + 1, len(series))):
                return i  # Convergence point found with absolute threshold
        elif rel_threshold is not None:
            if all(abs(series[j] - series[j - 1]) < rel_threshold * series[j - 1] for j in range(i + 1, len(series))):
                return i  # Convergence point found with relative threshold

    # If no convergence point is found
    return -1


def find_convergence_point_all_samples(df, abs_threshold=None, rel_threshold=None):
    grouped = df.groupby('lineKey') # Group by image

    convergence_points_all_samples = [
        find_convergence_point(group["lineVal"].values,
                               abs_threshold=abs_threshold, rel_threshold=rel_threshold) for _, group in grouped
    ]

    return convergence_points_all_samples


def get_speed_up_calculations(speed_ups_per_img):
    percentage_samples_reaching_max_score_earlier = sum(1 for x in speed_ups_per_img if x != 0) / len(speed_ups_per_img)
    total_samples_reaching_max_score_earlier = sum(1 for x in speed_ups_per_img if x != 0)
    p_test_binom = stats.binomtest(total_samples_reaching_max_score_earlier, n=len(speed_ups_per_img), p=0.5,
                                   alternative="greater")

    # Calculate the average speed up, given that there is a speed-up:
    average_speed_up_given_more_efficient = np.mean([x for x in speed_ups_per_img if x != 0])
    std_speed_up_given_more_efficient = np.std([x for x in speed_ups_per_img if x != 0])

    return percentage_samples_reaching_max_score_earlier, p_test_binom, average_speed_up_given_more_efficient, \
        std_speed_up_given_more_efficient


def df_to_numpy_array(df):
    """
    Converts the given DataFrame to a NumPy array of shape (300, 500).

    Parameters:
    - df (pandas.DataFrame): Input DataFrame with columns 'step', 'lineKey', and 'lineVal'.

    Returns:
    - numpy.ndarray: A 300x500 NumPy array.
    """
    # Pivot the DataFrame to reshape it
    pivoted_df = df.pivot(index='step', columns='lineKey', values='lineVal')

    # Convert the DataFrame to a NumPy array
    return pivoted_df.to_numpy()


def min_max_scale(array):
    # Flatten across iterations and samples to get the global min and max for each method
    min_val = np.min(array, axis=(0, 1), keepdims=True)  # Global min for each method
    max_val = np.max(array, axis=(0, 1), keepdims=True)  # Global max for each method

    # Apply min-max scaling
    return (array - min_val) / (max_val - min_val)


def clip_to_percentiles(array, lower_percentile=1, upper_percentile=95):
    """
    Clip the values of the array to the 1st and 99th percentiles.

    Parameters:
    - array: NumPy array of any shape
    - lower_percentile: The lower percentile (default is 1%)
    - upper_percentile: The upper percentile (default is 99%)

    Returns:
    - Clipped array with values constrained between the specified percentiles.
    """
    # Calculate the lower and upper percentile values
    lower_value = np.percentile(array, lower_percentile)
    upper_value = np.percentile(array, upper_percentile)

    # Clip the array to this range
    clipped_array = np.clip(array, lower_value, upper_value)

    return clipped_array


def normalize_scores(array_1, array_2, array_3, reference_array_idx=None,
                     apply_min_max_scale=False, apply_clipping=False):
    """
    Normalizes the three array scores
    """

    if apply_clipping:
        array_1 = clip_to_percentiles(array_1)
        array_2 = clip_to_percentiles(array_2)
        array_3 = clip_to_percentiles(array_3)

    # First normalize the scores for across each sample
    total_arrays = np.stack((array_1, array_2, array_3))

    # Min-max scale
    if apply_min_max_scale:
        total_arrays = min_max_scale(total_arrays)

    # Total arrays are now 3, 300, 500
    if reference_array_idx is None:
        max_score = np.max(total_arrays, axis=(0, 1))
    else:
        reference_array = total_arrays[reference_array_idx, :, :]
        max_score = np.max(reference_array, axis=0)

    if apply_min_max_scale:
        array_1 = total_arrays[0, :, :] / max_score
        array_2 = total_arrays[1, :, :] / max_score
        array_3 = total_arrays[2, :, :] / max_score
    else:
        array_1 /= max_score
        array_2 /= max_score
        array_3 /= max_score

    return array_1, array_2, array_3


def get_iterations_to_target(array, mean):
    mean_iterations_to_target = []
    std_iterations_to_target = []
    for score in mean:
        iterations_to_score = []
        for example in range(array.shape[1]):
            if np.any(array[:, example] >= score):
                iterations_to_score.append(np.where(array[:, example] >= score)[0][0])
            else:
                iterations_to_score.append(np.nan)
        mean_iterations_to_target.append(np.nanmean(iterations_to_score))
        std_iterations_to_target.append(np.nanstd(iterations_to_score))

    return np.array(mean_iterations_to_target), np.array(std_iterations_to_target)


def plot_iterations_across_scores(array_1, array_2, array_3, model, method, metric, type="mean",
                                  uncertainty=True, std_error=True, factor=1., save_path=None):
    """
    Plots the normalized scores across iterations for three methods.

    Parameters:
    - array_1: NumPy array of shape (300, 500) for Method A
    - array_2: NumPy array of shape (300, 500) for Method B
    - array_3: NumPy array of shape (300, 500) for Method C (if needed)
    """

    # Relabel the metric
    metric = "CP-Pixel" if metric =="cp_pixel" else "CP-L1"

    # Compute mean and std across runs for both methods
    mean_a = np.mean(array_1, axis=1)[::10]
    std_a = factor * (np.std(array_1, axis=1)[::10]/np.sqrt(array_1.shape[1])) if std_error \
        else np.std(array_1, axis=1)[::10]

    median_a = np.median(array_1, axis=1)[::10]
    q25_a = np.percentile(array_1, q=25, axis=1)[::10]
    q75_a = np.percentile(array_1, q=75, axis=1)[::10]

    mean_b = np.mean(array_2, axis=1)[::10]
    std_b = factor * (np.std(array_2, axis=1)[::10]/np.sqrt(array_2.shape[1])) if std_error \
        else np.std(array_2, axis=1)[::10]

    median_b = np.median(array_2, axis=1)[::10]
    q25_b = np.percentile(array_2, q=25, axis=1)[::10]
    q75_b = np.percentile(array_2, q=75, axis=1)[::10]

    # If you want to plot Method C as well
    mean_c = np.mean(array_3, axis=1)[::10]
    std_c = factor * (np.std(array_3, axis=1)[::10]/np.sqrt(array_3.shape[1])) if std_error \
        else np.std(array_3, axis=1)[::10]

    median_c = np.median(array_3, axis=1)[::10]
    q25_c = np.percentile(array_3, q=25, axis=1)[::10]
    q75_c = np.percentile(array_3, q=75, axis=1)[::10]

    # Define the iterations (1 to 300)
    iterations = np.arange(1, 301, 10)

    # Calculate mean and std for iterations to reach the target score
    mean_iterations_a, std_iterations_a = get_iterations_to_target(array_1, mean_a)
    mean_iterations_b, std_iterations_b = get_iterations_to_target(array_2, mean_b)
    mean_iterations_c, std_iterations_c = get_iterations_to_target(array_3, mean_c)

    # Create the plot
    plt.figure(figsize=(6, 4))

    if type == "IQR":
        # Method A plot with error bands
        plt.plot(median_a, iterations, label='StartGrad (ours)', linestyle='-', color=colors["sky_blue"])
        if uncertainty:
            plt.fill_betweenx(iterations, q25_a, q75_a, color=colors["sky_blue"], alpha=0.2)

        # Method B plot with error bands
        plt.plot(median_b, iterations, label='All-ones (baseline)', linestyle='--', color=colors["orange"])
        if uncertainty:
            plt.fill_betweenx(iterations, q25_b, q75_b, color=colors["orange"], alpha=0.2)

        # If you want to plot Method C
        plt.plot(median_c, iterations, label='Uniform', linestyle=':', color=colors["green"])
        if uncertainty:
            plt.fill_betweenx(iterations, q25_c, q75_c, color=colors["green"], alpha=0.2)

    else:
        # Method A plot with error bands
        plt.plot(mean_a, iterations, label='StartGrad (ours)', linestyle='-', color=colors["sky_blue"])
        if uncertainty:
            plt.fill_betweenx(iterations, mean_a - std_a, mean_a + std_a, color=colors["sky_blue"], alpha=0.2)

        # Method B plot with error bands
        plt.plot(mean_b, iterations, label='All-ones', linestyle='--', color=colors["orange"])
        if uncertainty:
            plt.fill_betweenx(iterations, mean_b - std_b, mean_b + std_b, color=colors["orange"], alpha=0.2)

        # If you want to plot Method C
        plt.plot(mean_c, iterations, label='Uniform', linestyle=':', color=colors["green"])
        if uncertainty:
            plt.fill_betweenx(iterations, mean_c - std_c, mean_c + std_c, color=colors["green"], alpha=0.2)

    # Show every 50th iteration on the x-axis
    plt.xticks(ticks=np.arange(0, 1, 0.10), fontsize=16)
    plt.yticks(fontsize=16)

    # Labels, legend, and grid
    plt.xlabel(f'$\\mathbf{{Normalized}}\\ \\mathbf{{{metric}}}\\ \\mathbf{{Score}}$', fontsize=16)
    # plt.xlabel(f'Normalized {metric} Score', fontsize=20)
    plt.ylabel('$\\mathbf{Iterations}$', fontsize=16)
    #plt.ylabel('Iterations', fontsize=20)
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(False)

    # Display the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"Time_score_{model}_{method}_{metric}.pdf"), dpi=300, format="pdf"
    )

    plt.close()


def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its content.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_iqm_and_se(data: list) -> tuple:
    """
    Calculate the interquartile mean (IQM) and standard error (SE) for a given list of values.

    Args:
        data (list): List of numerical values.

    Returns:
        tuple: IQM and SE of the data.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    data_iqm = [x for x in data if q1 <= x <= q3]
    iqm = np.mean(data_iqm)
    se = np.std(data_iqm) / np.sqrt(len(data_iqm))  # Standard error calculation
    return np.round(iqm, 3), np.round(se, 3)


def main(args):
    base_path_figures = os.path.join(FIGURES_PATH, args.method, str(args.seed), args.model_type)
    # Create directory if needed:
    create_directory(base_path_figures)

    base_path_results = os.path.join(RESULTS_PATH, args.method)

    # Load the data
    file_path_1 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{str(args.seed)}",
                               f"{args.metric}" + "_base.csv")
    file_path_2 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                               f"{args.metric}" + "_uniform.csv")
    file_path_3 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                               f"{args.metric}" + "_startgrad.csv")

    if args.get_comparison_gradient:
        file_path_ig = os.path.join(RESULTS_PATH, "quantitative_performance_measures_['IG']" + ".json")
        file_path_ig_qtf = os.path.join(RESULTS_PATH, "quantitative_performance_measures_['IG']_qtf" + ".json")

        file_path_grad_cam = os.path.join(RESULTS_PATH, "quantitative_performance_measures_['GradCAM']" + ".json")
        file_path_grad_cam_qtf = os.path.join(RESULTS_PATH, "quantitative_performance_measures_['GradCAM']_qtf" + ".json")

        # Load the JSON data
        ig_data = load_json(file_path_ig)
        ig_data_qtf = load_json(file_path_ig_qtf)

        grad_cam_data = load_json(file_path_grad_cam)
        grad_cam_data_qtf = load_json(file_path_grad_cam_qtf)

        ig_iqm, ig_se = calculate_iqm_and_se(ig_data['IG'])
        ig_iqm_qtf, ig_se_qtf = calculate_iqm_and_se(ig_data_qtf['IG'])

        grad_cam_iqm, grad_cam_se = calculate_iqm_and_se(grad_cam_data['GradCAM'])
        grad_cam_iqm_qtf, grad_cam_se_qtf = calculate_iqm_and_se(grad_cam_data_qtf['GradCAM'])

        # Load the data for the StartGrad initialization:
        data_startgrad = load_data(file_path_3)
        stats_startgrad_sg = calculate_performance_metric_by_step(data_startgrad, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        sg_metric_50 = stats_startgrad_sg[f"performance_{args.statistics}"][49].round(3)
        sg_metric_std_50 = stats_startgrad_sg["performance_std"][49].round(3)

        sg_metric_100 = stats_startgrad_sg[f"performance_{args.statistics}"][99].round(3)
        sg_metric_std_100 = stats_startgrad_sg["performance_std"][99].round(3)

        sg_metric_300 = stats_startgrad_sg[f"performance_{args.statistics}"][299].round(3)
        sg_metric_std_300 = stats_startgrad_sg["performance_std"][299].round(3)

        print(f"StartGrad {args.method} {args.metric} step: 50: IQM:{sg_metric_50} SE:{sg_metric_std_50}")
        print(f"StartGrad {args.method} {args.metric} step: 100: IQM:{sg_metric_100} SE:{sg_metric_std_100}")
        print(f"StartGrad {args.method} {args.metric} step: 300: IQM:{sg_metric_300} SE:{sg_metric_std_300}")

        # Now doing the same for QTF transform
        print(f"IG IQM (Min-max): {ig_iqm}, SE: {ig_se}")
        print(f"GradCAM IQM (Min-max): {grad_cam_iqm}, SE: {grad_cam_se}")
        print(f"IG IQM (QTF): {ig_iqm_qtf}, SE: {ig_se_qtf}")
        print(f"GradCAM IQM (QTF): {grad_cam_iqm_qtf}, SE: {grad_cam_se_qtf}")

    if args.retained_probability or args.retained_information:
        if args.retained_probability:
            file_path_4 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{str(args.seed)}",
                                       "retained_probability_" + "startgrad" + "_average" + ".csv")

            file_path_5 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{str(args.seed)}",
                                       "retained_probability_" + "uniform" + "_average" + ".csv")

        else:
            file_path_4 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                       f"{str(args.seed)}",
                                       "retained_information_l1_" + "startgrad" + "_average" + ".csv")

            file_path_5 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                       f"{str(args.seed)}",
                                       "retained_information_l1_" + "uniform" + "_average" + ".csv")
        # load the dataframe
        retained_prob_startgrad = load_data(file_path_4)
        retained_prob_uniform = load_data(file_path_5)

        combined_stats_prob = pd.concat([retained_prob_startgrad, retained_prob_uniform],
                                           keys=["StartGrad (ours)", "Uniform"],
                                           names=['Initialization'])

        plot_retained_prob_sparsity(
            combined_stats_prob,
            save_path=base_path_figures,
            retained_prob=args.retained_probability,
            statistics="average", #We have only average implemented and data available
            method=args.method,
            show_plot=args.show_plot,
        )

    if args.add_smoothgrad:
        file_path_sg = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                    f"{args.metric}" + "_startgrad_noisy.csv")
        df_startgrad_sg = load_data(file_path_sg)
        stats_startgrad_sg = calculate_performance_metric_by_step(df_startgrad_sg, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

    if args.add_lambda_l1:

        if args.method in ["waveletX"]:
            metric_1_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_1_10" + ".csv")

            metric_01_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_01_10" + ".csv")

            metric_10_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_10_10" + ".csv")

            key_list = [r"$\lambda_{1}=1$, $\lambda_{2}=10$",
                        r"$\lambda_{1}=0.1$, $\lambda_{2}=10$",
                        r"$\lambda_{1}=10$, $\lambda_{2}=10$",]

        elif args.method in ["shearletX"]:
            metric_1_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + ".csv")

            metric_01_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_01_2" + ".csv")

            metric_10_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_10_2" + ".csv")

            key_list = [r"$\lambda_{1}=1$, $\lambda_{2}=2$",
                        r"$\lambda_{1}=0.1$, $\lambda_{2}=2$",
                        r"$\lambda_{1}=10$, $\lambda_{2}=2$",]

        else:
            # In pixelRDE we only have lambda_l1 parameter, we still call it still with 10 lambda l2 but this is
            # but this is actually not very clean
            metric_1_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_1" + ".csv")

            metric_01_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_01" + ".csv")

            metric_10_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_10" + ".csv")

            key_list = [r"$\lambda_{1}=1$", r"$\lambda_{1}=0.1$", r"$\lambda_{1}=10$"]

        # load the dataframe
        df_1_10 = load_data(metric_1_10)
        df_01_10 = load_data(metric_01_10)
        df_10_10 = load_data(metric_10_10)

        stats_1_10 = calculate_performance_metric_by_step(df_1_10, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)
        stats_01_10 = calculate_performance_metric_by_step(df_01_10, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)
        stats_10_10 = calculate_performance_metric_by_step(df_10_10, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        combined_stats_lambda_l1 = pd.concat([stats_1_10, stats_01_10, stats_10_10],
                                           keys=key_list,
                                           names=['Initialization'])

        plot_performance_metric_by_step(
            combined_stats_lambda_l1,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            lambda_l1=args.add_lambda_l1,
            initialization=args.initialization,
            q=args.q,
            show_plot=args.show_plot,
        )

    if args.add_lambda_l2:

        if args.method in ["shearletX"]:
            metric_1_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                       f"{str(args.seed)}",
                                       f"{args.metric}" + f"_{args.initialization}" + "_1_20" + ".csv")

            metric_1_1 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                      f"{str(args.seed)}",
                                      f"{args.metric}" + f"_{args.initialization}" + "_1_2" + ".csv")

            metric_1_01 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                       f"{str(args.seed)}",
                                       f"{args.metric}" + f"_{args.initialization}" + "_1_02" + ".csv")

            keys_list = [r"$\lambda_{1}=1$, $\lambda_{2}=20$",
                         r"$\lambda_{1}=1$, $\lambda_{2}=2$",
                         r"$\lambda_{1}=1$, $\lambda_{2}=0.2$"]

        else:
            metric_1_10 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_1_10" + ".csv")

            metric_1_1 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_1_1" + ".csv")

            metric_1_01 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                        f"{args.metric}" + f"_{args.initialization}" + "_1_01" + ".csv")

            keys_list = [r"$\lambda_{1}=1$, $\lambda_{2}=10$",
                         r"$\lambda_{1}=1$, $\lambda_{2}=1$",
                         r"$\lambda_{1}=1$, $\lambda_{2}=0.1$"]

        # load the dataframe
        df_1_10 = load_data(metric_1_10)
        df_1_1 = load_data(metric_1_1)
        df_1_01 = load_data(metric_1_01)

        stats_1_10 = calculate_performance_metric_by_step(df_1_10, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)
        stats_1_1 = calculate_performance_metric_by_step(df_1_1, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)
        stats_1_01 = calculate_performance_metric_by_step(df_1_01, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        combined_stats_lambda_l2 = pd.concat([stats_1_10, stats_1_1, stats_1_01],
                                           keys=keys_list,
                                           names=['Initialization'])

        plot_performance_metric_by_step(
            combined_stats_lambda_l2,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            lambda_l2=args.add_lambda_l2,
            initialization=args.initialization,
            q=args.q,
        )

    if args.add_shuffle:
        # Load the shuffled baseline startgrad data
        file_path_shuffle = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                         f"{str(args.seed)}", f"{args.metric}" + "_startgrad_baseline_shuffle.csv")
        df_baseline_shuffle = load_data(file_path_shuffle)
        df_baseline_shuffle['lineVal'] = df_baseline_shuffle['lineVal'].astype(float)
        stats_baseline_shuffle = calculate_performance_metric_by_step(df_baseline_shuffle, statistics=args.statistics,
                                                                      q=args.q, bootstrap=args.bootstrap)

        # Load the shuffled uniform data
        file_path_uniform_shuffle = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                                 f"{str(args.seed)}", f"{args.metric}" + "_uniform_shuffle.csv")
        df_uniform_shuffle = load_data(file_path_uniform_shuffle)
        df_uniform_shuffle['lineVal'] = df_uniform_shuffle['lineVal'].astype(float)
        stats_uniform_shuffle = calculate_performance_metric_by_step(df_uniform_shuffle, statistics=args.statistics,
                                                                      q=args.q, bootstrap=args.bootstrap)

        # Load the shuffled startgrad data
        file_path_startgrad_shuffle = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                                  f"{str(args.seed)}", f"{args.metric}" + "_startgrad_shuffle.csv")
        df_startgrad_shuffle = load_data(file_path_startgrad_shuffle)
        df_startgrad_shuffle['lineVal'] = df_startgrad_shuffle['lineVal'].astype(float)
        stats_startgrad_shuffle = calculate_performance_metric_by_step(df_startgrad_shuffle, statistics=args.statistics,
                                                                      q=args.q, bootstrap=args.bootstrap)

        if args.add_adversarial:
            # Load the adversarial data for StartGrad
            file_path_adversarial = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                                  f"{str(args.seed)}", f"{args.metric}" + "_startgrad_adversarial.csv")
            df_startgrad_adversarial = load_data(file_path_adversarial)
            df_startgrad_adversarial['lineVal'] = df_startgrad_adversarial['lineVal'].astype(float)
            stats_startgrad_adversarial = calculate_performance_metric_by_step(df_startgrad_adversarial,
                                                                               statistics=args.statistics,
                                                                               q=args.q, bootstrap=args.bootstrap)

        if args.add_adversarial:
            combined_stats_shuffle = pd.concat([stats_baseline_shuffle, stats_uniform_shuffle,
                                                stats_startgrad_shuffle,
                                                stats_startgrad_adversarial],
                                                keys=["StartGrad (baseline)", "Uniform",
                                                      "StartGrad (uninformative gradients)",
                                                      "StartGrad (adversarial gradients)"],
                                                names=['Initialization'])

        else:
        # Combine stats for plotting
            combined_stats_shuffle = pd.concat([stats_baseline_shuffle, stats_uniform_shuffle, stats_startgrad_shuffle],
                                                keys=["StartGrad (baseline)", "Uniform",
                                                      "StartGrad (uninformative gradients)"],
                                                names=['Initialization'])

        # Plot performance metrics for shuffle
        plot_performance_metric_by_step(
            combined_stats_shuffle,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            shuffle=True,
            q=args.q,
            show_plot=args.show_plot,
        )

    if args.add_min_max_comparison:
        file_path_min_max = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                          f"{str(args.seed)}", f"{args.metric}" + "_min_max_saliency.csv")
        file_path_startgrad_2 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                          f"{str(args.seed)}", f"{args.metric}" + "_startgrad_2.csv")

        df_startgrad_min_max = load_data(file_path_min_max)
        df_startgrad_2 = load_data(file_path_startgrad_2)

        stats_startgrad_min_max = calculate_performance_metric_by_step(df_startgrad_min_max, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        stats_startgrad_2 = calculate_performance_metric_by_step(df_startgrad_2, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        combined_stats_min_max = pd.concat([stats_startgrad_2,
                                          stats_startgrad_min_max
                                          ],
                                         keys=["StartGrad (baseline)",
                                               "StartGrad (min-max scaling)",
                                               ],
                                         names=['Initialization'])

        plot_performance_metric_by_step(
            combined_stats_min_max,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            min_max_comparison=True,
            q=args.q,
            show_plot=args.show_plot,
        )

    if args.add_noisy:
        file_path_4 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",  f"{str(args.seed)}",
                                   f"{args.metric}" + "_startgrad_noisy.csv")
        df_startgrad_noisy = load_data(file_path_4)
        df_startgrad_noisy['lineVal'] = df_startgrad_noisy['lineVal'].astype(float)
        stats_startgrad_noisy = calculate_performance_metric_by_step(df_startgrad_noisy, statistics=args.statistics,
                                                                     q=args.q, bootstrap=args.bootstrap)

        file_path_5 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                    f"{str(args.seed)}", f"{args.metric}" + "_startgrad_noisy_005.csv")
        df_startgrad_noisy_005 = load_data(file_path_5)
        df_startgrad_noisy_005['lineVal'] = df_startgrad_noisy_005['lineVal'].astype(float)
        stats_startgrad_noisy_005 = calculate_performance_metric_by_step(df_startgrad_noisy_005,
                                                                         statistics=args.statistics,
                                                                         q=args.q, bootstrap=args.bootstrap)

        file_path_6 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}",
                                    f"{str(args.seed)}", f"{args.metric}" + "_startgrad_noisy_02.csv")
        df_startgrad_noisy_02 = load_data(file_path_6)
        df_startgrad_noisy_02['lineVal'] = df_startgrad_noisy_02['lineVal'].astype(float)
        stats_startgrad_noisy_02 = calculate_performance_metric_by_step(df_startgrad_noisy_02,
                                                                         statistics=args.statistics,
                                                                         q=args.q, bootstrap=args.bootstrap)

    df_ones = load_data(file_path_1)
    df_unif = load_data(file_path_2)
    df_startgrad = load_data(file_path_3)

    # Convert the 'lineVal' column to float, if not already
    df_ones['lineVal'] = df_ones['lineVal'].astype(float)
    df_unif['lineVal'] = df_unif['lineVal'].astype(float)
    df_startgrad['lineVal'] = df_startgrad['lineVal'].astype(float)

    # These are of shape 300, 500 (time_step, number_samples)
    startgrad_numpy = df_to_numpy_array(df_startgrad)
    base_ones_numpy = df_to_numpy_array(df_ones)
    base_uniform_numpy = df_to_numpy_array(df_unif)

    # Normalized scores
    startgrad_norm_score, base_ones_norm_score, base_uniform_norm_score = normalize_scores(
        startgrad_numpy, base_ones_numpy, base_uniform_numpy)

    # Plot the iterations across scores:
    plot_iterations_across_scores(startgrad_norm_score, base_ones_norm_score, base_uniform_norm_score,
                                  model=args.model_type,
                                  method=args.method, metric=args.metric, save_path=base_path_figures)

    # # Calculate bootstrapped statistics by step
    stats_ones = calculate_performance_metric_by_step(df_ones, statistics=args.statistics,
                                                      q=args.q, bootstrap=args.bootstrap)
    stats_unif = calculate_performance_metric_by_step(df_unif, statistics=args.statistics,
                                                      q=args.q, bootstrap=args.bootstrap)
    stats_startgrad = calculate_performance_metric_by_step(df_startgrad,
                                                           statistics=args.statistics,
                                                           q=args.q, bootstrap=args.bootstrap)

    # Threshold to match
    baseline_threshold = stats_ones[f"performance_{args.statistics}"].values[-1]

    startgrad_convergence_rates = calculate_convergence_rates(df_startgrad, baseline_threshold)
    uniform_convergence_rates = calculate_convergence_rates(df_unif, baseline_threshold)

    # Create a DataFrame to store convergence rates and corresponding initialization methods
    convergence_data = pd.DataFrame({
        'Convergence Rate': startgrad_convergence_rates + uniform_convergence_rates,
        'Initialization': ['StartGrad (ours)'] * len(startgrad_convergence_rates) +
                          ['Uniform'] * len(uniform_convergence_rates)
    })

    # Combine all stats for plotting
    combined_stats = pd.concat([stats_ones, stats_unif, stats_startgrad],
                               keys=["All-ones \n(baseline)", "Uniform", "StartGrad (ours)"],
                               names=['Initialization'])

    plot_performance_metric_by_step(
        combined_stats,
        save_path=base_path_figures,
        model=args.model_type,
        metric=args.metric,
        method=args.method,
        statistics=args.statistics,
        q=args.q,
    )

    if args.add_smoothgrad:
        combined_stats_sg = pd.concat([stats_startgrad,
                                          stats_startgrad_sg,
                                          ],
                                   keys=["StartGrad (baseline)",
                                         "StartGrad + SmoothGrad"
                                         ],
                                   names=['Initialization'])

        plot_performance_metric_by_step(
            combined_stats_sg,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            q=args.q,
            noisy=False,
            smoothgrad=True,
        )

    if args.add_noisy:
        combined_stats_noisy = pd.concat([stats_startgrad,
                                          stats_startgrad_noisy,
                                          stats_startgrad_noisy_005,
                                          stats_startgrad_noisy_02,
                                          ],
                                   keys=["StartGrad \n$\sigma^{2} = 0.00$",
                                         "StartGrad \n$\sigma^{2} = 0.01$",
                                         "StartGrad \n$\sigma^{2} = 0.05$",
                                         "StartGrad \n$\sigma^{2} = 0.20$"
                                         ],
                                   names=['Initialization'])

        plot_performance_metric_by_step(
            combined_stats_noisy,
            save_path=base_path_figures,
            model=args.model_type,
            metric=args.metric,
            method=args.method,
            statistics=args.statistics,
            q=args.q,
            noisy=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and plot performance metrics.')
    parser.add_argument('--method', type=str, help='Method name', default="shearletX")
    parser.add_argument('--seed', type=int, help='Random seed', default=123)
    parser.add_argument('--model_type', type=str, help='Model type', default="resnet18")
    parser.add_argument('--metric', type=str, help='Metric to evaluate', default="cp_pixel",
                        choices=("cp_pixel", "cp_l1"))
    parser.add_argument("--statistics", type=str, help='which statistics to calculate. mean, median, quantile,'
                                                       'median, interquartile_mean',
                        default="interquartile_mean")
    parser.add_argument("--q", type=float, help="which quantile to use",
                        default=0.50)
    parser.add_argument("--show_plot", action="store_true", help="Boolean. If set, it will show all generated plots" )
    parser.add_argument("--add_smoothgrad", action="store_true", help="Add smoothgrad version")
    parser.add_argument("--add_noisy", action="store_true", help="Add the noisy version of the startgrad version")
    parser.add_argument("--add_min_max_comparison", action="store_true", help="Show the comparison min-max version")
    parser.add_argument("--bootstrap", action="store_true", help="If we want to bootstrap the metrics")
    parser.add_argument("--add_shuffle", action="store_true", help="If we want to add the shuffled robustness results")
    parser.add_argument("--add_adversarial", action="store_true", help="If we want to the "
                                                                       "adversarial robustness results")
    parser.add_argument("--add_lambda_l1", action="store_true",
                        help="If we want to do the ablation study for the varying lambda 1")
    parser.add_argument("--add_lambda_l2", action="store_true",
                        help="If we want to do the ablation study for the varying lambda 2")
    parser.add_argument("--retained_probability", action="store_true",
                        help="If we want to do the analysis for retained probability")
    parser.add_argument("--retained_information", action="store_true",
                        help="If we want to do the analysis for retained information")
    parser.add_argument("--initialization", default="startgrad",
                        help="which initialization to choose for lambda 2 or lambda 1 ablation"
                             "choices: base, startgrad, uniform")
    parser.add_argument("--get_comparison_gradient", action="store_true",
                        help="If we want to use the comparison with posthoc")

    args = parser.parse_args()
    args.bootstrap = True
    args.show_plot = True
    pl.seed_everything(args.seed)
    main(args)

