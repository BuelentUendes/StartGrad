import os
import csv
import json
import argparse
from scipy import stats
import numpy as np
import pytorch_lightning as pl
from utils.general.helper_path import RESULTS_PATH
from utils.general.helper import create_directory


def parse_methods(arg):
    items = arg.split(",")
    # Get rid of whitespace
    items = [item.strip() for item in items]

    return items


def get_metric_results(method, seed, model, results_path=RESULTS_PATH, metric="cp_l1"):

    # Get the name of the subdirectory based on the method
    subdirectory = method.split("_")[0]
    try:
        suffix = method.split("_")[1]
    except:
        suffix = "base"

    if suffix == "saliency":
        suffix = "startgrad"

    if suffix == "uniform":
        suffix = "uniform"

    file_name = f"{metric}_{suffix}.csv"
    saved_results_path = os.path.join(results_path, "vision", subdirectory, model, str(seed))
    metric_results = []

    with open(os.path.join(saved_results_path, file_name), mode='r') as f:
        csv_reader = csv.DictReader(f)  # Reads the file as a list of dictionaries
        for row in csv_reader:
            metric_results.append(row)

    return metric_results


def create_save_directory(method, seed, model, results_path=RESULTS_PATH):
    subdirectory = method.split("_")[0]
    saved_results_path = os.path.join(results_path, "vision", subdirectory, model, str(seed), "statistical_results")
    create_directory(saved_results_path)

    return saved_results_path


def get_performance_difference(performance_values_1, performance_values_2, ):
    results_array_1 = np.asarray(performance_values_1)
    results_array_2 = np.asarray(performance_values_2)

    performance_difference = results_array_1 - results_array_2
    return performance_difference


def get_wilcoxon_statistical_result(difference_values, alternative="greater", correction=True):
    _, p_signed_test_result = stats.wilcoxon(difference_values, alternative=alternative, correction=correction)
    return p_signed_test_result


def get_t_test_statistical_result(difference_value_1, difference_value_2, alternative="greater"):
    # Perform a two-sided paired t-test
    t_stat, p_value = stats.ttest_rel(difference_value_1, difference_value_2)

    # Convert the two-sided p-value to a one-sided p-value if alternative is greater
    if alternative == "greater":
        if t_stat > 0:
            p_value /= 2
        else:
            p_value = 1 - (p_value / 2)

    return p_value


def get_wilcoxon_statistical_results_over_iterations(difference_values, alternative="greater", correction=True):
    # difference values is of shape images X iterations

    number_iterations = difference_values.shape[1]
    p_values_over_iterations = [
        get_wilcoxon_statistical_result(difference_values[:, iteration], alternative=alternative, correction=correction)
        for iteration in range(number_iterations)
    ]

    return p_values_over_iterations


def restructure_performance_metrics(performance_metric):
    new_performance_metric = {}
    for performance_dict in performance_metric:
        try:
            new_performance_metric[performance_dict["lineKey"]].append(float(performance_dict["lineVal"]))
        except:
            new_performance_metric[performance_dict["lineKey"]] = [float(performance_dict["lineVal"])]
    
    return new_performance_metric


def main(args):

    performance_metric_results = {
        method: get_metric_results(method, args.seed, args.pretrained_model, metric=args.metric)
        for method in args.methods
    }

    performance_metric_1 = performance_metric_results[args.methods[0]]
    performance_metric_2 = performance_metric_results[args.methods[1]]
    
    performance_metric_1 = restructure_performance_metrics(performance_metric_1)
    performance_metric_2 = restructure_performance_metrics(performance_metric_2)

    # Step 1: Convert dictionaries to a 2D NumPy array (500 images x 300 iterations)
    method_1_array = np.array([performance_metric_1[key] for key in performance_metric_1.keys()])  # shape: (500, 300)
    method_2_array = np.array([performance_metric_2[key] for key in performance_metric_2.keys()])  # shape: (500, 300)

    # Get overall median performance difference
    median_array_1 = np.median(method_1_array, axis=0)
    median_array_2 = np.median(method_2_array, axis=0)

    overall_median_difference = median_array_1 - median_array_2

    # Step 2: Calculate the differences across all images and iterations (vectorized)
    differences = method_1_array - method_2_array  # shape: (500, 300)

    # Step 3: Transpose the array to get 300 iterations as rows and 500 images as columns
    transposed_differences = differences.T

    if args.apply_filter:
        # Keep only a specific range
        lower_percentile = np.percentile(transposed_differences, args.lower_threshold, axis=1, keepdims=True)
        upper_percentile = np.percentile(transposed_differences, args.upper_threshold, axis=1, keepdims=True)

        # 2. Filter to get rid of upper and lower percentile values of transposed differences and single differences
        mask = (transposed_differences >= lower_percentile) & (transposed_differences <= upper_percentile)

        filtered_values = np.where(mask, transposed_differences, np.nan)
        filtered_values_1 = np.where(mask.T, method_1_array, np.nan)
        filtered_values_2 = np.where(mask.T, method_2_array, np.nan)

        # Remove NaN values (flatten the array and remove NaNs)
        transposed_differences = filtered_values[~np.isnan(filtered_values)].reshape(transposed_differences.shape[0], -1)
        method_1_array = filtered_values_1[~np.isnan(filtered_values_1)].reshape(-1, method_1_array.shape[1])
        method_2_array = filtered_values_2[~np.isnan(filtered_values_2)].reshape(-1, method_2_array.shape[1])

    # Calculate average delta:
    average_delta_per_step = np.mean(transposed_differences, axis=1)

    # Calculate median performance
    median_delta_per_step = np.median(transposed_differences, axis=1)

    # Calculate std delta
    std_delta_per_step = np.std(transposed_differences, axis=1)

    # Calculate the percentage of positive values along dimension 1
    positive_percentage = np.mean(transposed_differences > 0, axis=1) * 100
    # Number of positive differences

    if args.test == "t_test":
        p_values = [
            get_t_test_statistical_result(value_1, value_2, alternative="greater")
            for value_1, value_2 in zip(method_1_array, method_2_array)
        ]

        p_value_normality = [
            stats.normaltest(transposed_differences[iteration])[1]
            for iteration in range(transposed_differences.shape[0])
        ]

    else:
        p_values = [
            get_wilcoxon_statistical_result(transposed_differences[iteration],
                                            alternative=args.hypothesis, correction=True)
            for iteration in range(transposed_differences.shape[0])
        ]

    final_values = {
        str(iteration): {
            "median delta per step": median_delta_per_step[iteration],
            "positive percentage": positive_percentage[iteration],
            f"p_value {args.test}": p_values[iteration],
            f"p_value_normality": p_value_normality[iteration] if args.test =="t_test" else -1,
        }
        for iteration in ([49, 99, -1] if not args.save_all_steps else range(transposed_differences.shape[0]))
    }

    # File_name
    file_name = str(args.methods[0]) + "_vs_" + str(args.methods[1]) + "_" + args.hypothesis + "_" \
                + f"{args.metric}" + ".json"

    # Create the directory to save the results and get the save_path:
    save_path = create_save_directory(args.methods[0], args.seed, args.pretrained_model, results_path=RESULTS_PATH)

    with open(os.path.join(save_path, file_name), "w") as file:
        json.dump(final_values, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",
                        help="Indicate for which methods you want to get the significance test. Need to be 2!",
                        type=parse_methods, default="shearletX_saliency, shearletX")
    parser.add_argument("--metric", default="cp_l1", help="which metric to use for the statistical tests")
    parser.add_argument('--pretrained_model', type=str,
                        choices=('resnet18', 'vgg16',
                                 'swin_t', 'vit_base'),
                        help='Please specify which trained model you want to use for running CartoonX',
                        default='resnet18')
    parser.add_argument("--seed", help="From which seed one wants to get the results from", default=123, type=int)
    parser.add_argument("--hypothesis", help="Defines the alternative hypothesis for the Wilcoxon signed rank test",
                        default="greater", type=str, choices=("two-sided", "greater", "less"))
    parser.add_argument("--apply_filter", action="store_true")
    parser.add_argument("--upper_threshold", help="upper threshold when we apply a filter to remove outliers",
                        type=int, default=99.0)
    parser.add_argument("--lower_threshold", help="lower threshold when we apply a filter to remove outliers",
                        type=int, default=1.0)
    parser.add_argument("--test", help="which test to run. Choices (t_test, wilcoxon)", default="wilcoxon")
    parser.add_argument("--save_all_steps", action="store_true")

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    main(args)





