# Main script to run the main convergence calculations

import argparse
from typing import Optional
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from multiprocessing import Pool, cpu_count
from utils.general.helper_path import RESULTS_PATH


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
            if statistics == "mean":
                # Check this, as we remove 99 and 1 of the outlier here per step, check with average performance calculation!
                upper_threshold_value = data.quantile(0.99)
                lower_threshold_value = data.quantile(0.01)
                data = data[data >= lower_threshold_value]
                data = data[data <= upper_threshold_value]
                performance_metric = data.mean()
                print(performance_metric)
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


def bootstrap_samples(
        data: np.ndarray,
        n_samples: Optional[int] = None,
        random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Perform bootstrapping on the samples (columns) of a 2D array and return a new set of bootstrapped samples.

    Parameters:
    -----------
    data : np.ndarray
        The input 2D array of shape (iterations, samples) to be bootstrapped.
    n_samples : Optional[int], optional
        The number of bootstrapped samples (columns) to select.
        If None, the number of samples will be the same as the original data. Default is None.
    random_seed : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    np.ndarray
        A bootstrapped array with the same number of iterations but with bootstrapped samples.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get the original number of samples
    orig_samples = data.shape[1]

    # Use original number of samples if not specified
    if n_samples is None:
        n_samples = orig_samples

    # Generate random indices for bootstrapping (with replacement) across the samples
    sample_indices = np.random.randint(0, orig_samples, size=n_samples)

    # Return the bootstrapped data by selecting the random columns (samples)
    return data[:, sample_indices]


def convert_array_to_df(data: np.ndarray) -> pd.DataFrame:
    """
    Convert a 2D numpy array of shape (iterations, samples) into a DataFrame
    with columns ['step', 'lineKey', 'lineVal'].

    Parameters:
    -----------
    data : np.ndarray
        The input 2D array (iterations, samples) to be converted.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with 'step', 'lineKey', and 'lineVal' columns.
    """
    # Get the number of iterations (rows) and samples (columns)
    n_iterations, n_samples = data.shape

    # Create a DataFrame with flattened step and sample (lineKey)
    df = pd.DataFrame({
        'step': np.repeat(np.arange(n_iterations), n_samples),  # Repeats iteration number
        'lineKey': np.tile([f'random_image_{i}' for i in range(n_samples)], n_iterations),  # Assigns lineKey
        'lineVal': data.flatten()  # Flattens the 2D array into a 1D array of values
    })

    return df


def get_reference_value_and_idx(reference_method, statistics, reference_step, reference_step_max=False):
    if reference_step_max:  # Then we look for the maximum
        baseline_threshold_idx = reference_method[f"performance_{statistics}"].idxmax()
    else:
        baseline_threshold_idx = max(0, int(reference_step * len(reference_method)) - 1)

    # Retrieve the corresponding performance metric at index threshold step
    baseline_threshold_value = reference_method[f"performance_{statistics}"][baseline_threshold_idx]

    return baseline_threshold_value, baseline_threshold_idx


def bootstrap_sample_worker(startgrad_numpy, base_ones_numpy, base_uniform_numpy, statistics,
                            reference, reference_step, use_reference_step_max, q, count):
    """
    Worker function to perform bootstrapping and statistics calculation for a single sample.
    This is designed to run in parallel across multiple processes.

    Args:
        args: A tuple containing the data and necessary arguments for bootstrapping.

    Returns:
        A tuple of (speed_up, (step_idx, baseline_threshold_idx), verification_result)
    """
    print(f"Progress sample {count}")

    # Perform bootstrapping
    startgrad_numpy_bootstrapped = bootstrap_samples(startgrad_numpy, random_seed=count)
    base_ones_numpy_bootstrapped = bootstrap_samples(base_ones_numpy, random_seed=count)
    base_uniform_numpy_bootstrapped = bootstrap_samples(base_uniform_numpy, random_seed=count)

    # Convert bootstrapped samples into DataFrames
    df_startgrad_bootstrapped = convert_array_to_df(startgrad_numpy_bootstrapped)
    df_ones_bootstrapped = convert_array_to_df(base_ones_numpy_bootstrapped)
    df_unif_bootstrapped = convert_array_to_df(base_uniform_numpy_bootstrapped)

    # Calculate performance metrics for each bootstrapped DataFrame
    stats_ones = calculate_performance_metric_by_step(df_ones_bootstrapped, statistics=statistics, q=q)
    stats_unif = calculate_performance_metric_by_step(df_unif_bootstrapped, statistics=statistics, q=q)
    stats_startgrad = calculate_performance_metric_by_step(df_startgrad_bootstrapped, statistics=statistics, q=q)

    # Select reference method
    reference_method = stats_ones.copy() if reference == "all_ones" else stats_unif.copy()
    baseline_threshold_value, baseline_threshold_idx = get_reference_value_and_idx(reference_method,
                                                                                   statistics, reference_step,
                                                                                   use_reference_step_max)

    # Find the step where startgrad surpasses the baseline threshold`
    # Sometimes we do not reach the baseline_threshold, then we set it to equal 299 (final iteration step)
    if len(stats_startgrad[stats_startgrad[f'performance_{statistics}'] >= baseline_threshold_value].index) > 0:
        idx = stats_startgrad[stats_startgrad[f'performance_{statistics}'] >= baseline_threshold_value].index[0]
    else:
        idx = len(stats_ones) - 1

    #idx = stats_startgrad[stats_startgrad[f'performance_{statistics}'] >= baseline_threshold_value].index[0]
    # Calculate speed up
    speed_up = (baseline_threshold_idx / idx)

    # Verify if the performance consistently surpasses the baseline after the index
    verification = int((stats_startgrad[f'performance_{statistics}'].iloc[idx:] > baseline_threshold_value).all())

    return speed_up, idx, baseline_threshold_idx, verification


def main(args):
    base_path_results = os.path.join(RESULTS_PATH, "vision", args.method)

    # Load the data
    file_path_1 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{args.seed}", f"{args.metric}" + "_base.csv")
    file_path_2 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{args.seed}", f"{args.metric}" + "_uniform.csv")
    file_path_3 = os.path.join(RESULTS_PATH, "vision", f"{args.method}", f"{args.model_type}", f"{args.seed}", f"{args.metric}" + "_startgrad.csv")

    df_ones = load_data(file_path_1)
    df_unif = load_data(file_path_2)
    df_startgrad = load_data(file_path_3)

    # Convert the 'lineVal' column to float, if not already
    df_ones['lineVal'] = df_ones['lineVal'].astype(float)
    df_unif['lineVal'] = df_unif['lineVal'].astype(float)
    df_startgrad['lineVal'] = df_startgrad['lineVal'].astype(float)

    # Convert DataFrames to NumPy arrays
    startgrad_numpy = df_to_numpy_array(df_startgrad)
    base_ones_numpy = df_to_numpy_array(df_ones)
    base_uniform_numpy = df_to_numpy_array(df_unif)

    speed_ups_samples = []
    steps_need_to_reach_baseline = []

    if args.bootstrap:
        # Use multiprocessing Pool for parallel execution
        i = 0
        with Pool(processes=cpu_count()) as pool:
            i += 1
            # Prepare the arguments to be passed to each worker
            worker_args = [(startgrad_numpy, base_ones_numpy, base_uniform_numpy, args.statistics, args.reference,
                            args.reference_step, args.use_reference_step_max, args.q, count)
                           for count in range(args.bootstrap_samples)]
            # Map the worker function across multiple processes
            results = pool.starmap(bootstrap_sample_worker, worker_args)
            # Unpack the results
            for speed_up, step_idx, baseline_threshold_idx, verification in results:
                speed_ups_samples.append(speed_up)
                steps_need_to_reach_baseline.append(step_idx)

    else:
        # Non-bootstrap case (remains the same as original code)
        stats_ones = calculate_performance_metric_by_step(df_ones, statistics=args.statistics, q=args.q)
        stats_unif = calculate_performance_metric_by_step(df_unif, statistics=args.statistics, q=args.q)
        stats_startgrad = calculate_performance_metric_by_step(df_startgrad, statistics=args.statistics, q=args.q)

        reference_method = stats_ones if args.reference == "all_ones" else stats_unif
        baseline_threshold_value, baseline_threshold_idx = get_reference_value_and_idx(reference_method,
                                                                                       args.statistics,
                                                                                       args.reference_step,
                                                                                       args.use_reference_step_max,
                                                                                       )

        idx = stats_startgrad[stats_startgrad[f'performance_{args.statistics}'] >= baseline_threshold_value].index[0]
        speed_up = (baseline_threshold_idx / idx)
        speed_ups_samples.append(speed_up)
        steps_need_to_reach_baseline.append((idx, baseline_threshold_idx))


    # Now calculate the average speed-ups and standard error
    if args.bootstrap:
        assert len(speed_ups_samples) == args.bootstrap_samples, "Something went wrong with the bootstrapping!"

    average_speed_up = np.mean(speed_ups_samples)
    average_steps_to_reach_baseline = np.mean(steps_need_to_reach_baseline)
    std_steps_to_reach_baseline = np.std(steps_need_to_reach_baseline)
    std_error_speed_up = np.std(speed_ups_samples) if len(speed_ups_samples) > 1 else 0.

    print(
        f"Average speed up for: "
        f"\n{args.statistics} and reference {args.reference} max_setting: {args.use_reference_step_max}: "
        f"{average_speed_up}, {std_error_speed_up}, average steps required: {average_steps_to_reach_baseline} {std_steps_to_reach_baseline}"
    )

    with open(os.path.join(base_path_results, f"convergence_average_results_{args.metric}.txt"), "a") as f:
        f.write(
            f"\nAverage speed up for method: {args.method} model: {args.model_type} max_setting: {args.use_reference_step_max}"
            f"\n{args.statistics} and reference {args.reference}: speed-up: {average_speed_up}, std: {std_error_speed_up}, "
            f"average steps required: average_steps_required: {average_steps_to_reach_baseline}, std: {std_steps_to_reach_baseline}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process convergence calculations.')
    parser.add_argument('--method', type=str, help='Method name', default="shearletX")
    parser.add_argument('--seed', type=int, help='Random seed', default=123)
    parser.add_argument('--model_type', type=str, help='Model type', default="resnet18")
    parser.add_argument('--metric', type=str, help='Metric to evaluate', default="cp_pixel")
    parser.add_argument("--statistics", type=str, help='which statistics to calculate. mean, median, quantile,'
                                                       'interquartile_mean',
                        default="interquartile_mean")
    parser.add_argument("--q", type=float, help="which quantile to use if 'quantile' is set at statistics",
                        default=0.50)
    parser.add_argument("--bootstrap", action="store_true", help="If we want to bootstrap the metrics")
    parser.add_argument("--bootstrap_samples", type=int, default=250)
    parser.add_argument("--reference", default="uniform", help="Reference initialization for comparison")
    parser.add_argument("--reference_step", help="At what iteration step to take the reference comparison threshold, "
                                                    "between 0 and 1. 1 indicates last step, 0 first step.", default=1)
    parser.add_argument("--use_reference_step_max", action="store_true",
                        help="If set, we check use as the reference_step the idx at which the performance "
                             "is maximized across all iteration steps ")

    args = parser.parse_args()
    args.bootstrap = True
    pl.seed_everything(args.seed)
    main(args)
