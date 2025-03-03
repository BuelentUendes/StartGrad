import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from utils.general.helper_path import PERFORMANCE_TRADEOFF_PATH, FIGURES_PATH


def load_json_from_path(iteration_number=10):
    # Construct the file name using the iteration number
    file_name = f"summary_faithfulness_scores_number_{iteration_number}.json"

    # Build the full path to the file
    file_path = os.path.join(PERFORMANCE_TRADEOFF_PATH, file_name)

    # Open the JSON file and load its contents
    with open(file_path, mode='r') as f:
        data = json.load(f)  # Load JSON data into a Python dictionary or list, depending on file structure

    return data


def load_execution_time(method="IG", statistics="mean"):

    file_name = "Execution_time_" + method + ".txt"  # Ensure the correct file extension
    file_path = os.path.join(PERFORMANCE_TRADEOFF_PATH, file_name)  # Combine directory and file name

    execution_times = []

    # Open the file and read the execution times
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Convert each line to a float and add to the list (if it's a valid number)
                execution_times.append(float(line.strip()))
            except ValueError:
                pass
                # Skip the line if it can't be converted to a float
                #print(f"Skipping invalid line: {line.strip()}")

    final_time = np.mean(execution_times) if statistics == "mean" else np.sum(execution_times)

    return final_time


def get_attribution_category(method_name):

    white_box_methods = ["IG", "SG", "SmoothGrad", "Saliency", "VarGrad", "SquareGrad", "GradInput", "IntegratedGradients"]

    if method_name in white_box_methods:
        return "Gradient-based"

    return "Mask-based"


def adjust_method_names(method_name):
    names = {
        "GradInput": "Gradient\nInput",
        "SG": "SmoothGrad",
        "SmoothGrad": "SmoothGrad",
        "IG": "Integrated\nGradients",
        "IntegratedGradients": "Integrated\nGradients",
        "pixelRDE": "PixelMask",
        "waveletX": "WaveletX",
        "shearletX": "ShearletX",
        "SquareGrad": "SquareGrad",
        "VarGrad": "VarGrad",
        "Saliency": "Saliency"
    }

    adjusted_name = names[method_name]
    return adjusted_name


# Sample data for replication
def get_sample_data(iteration_number=10, statistics="mean"):
    """Returns sample data for the plot with faithfulness and execution time."""
    summary_statistics = load_json_from_path(iteration_number=iteration_number)

    # ToDo: Align the execution times names!
    data = {
        "methods": [method for method in summary_statistics.keys() if method != "random baseline"],
        "faithfulness": [value["average"] for method, value in summary_statistics.items() if method != "random baseline"],
    }

    data["categories"] = [get_attribution_category(method) for method in data["methods"]]
    data["execution_time"] = [
        load_execution_time(method=method, statistics=statistics) for method in data["methods"]
    ]

    # data = {
    #     "methods": ['Gradient Input', 'Saliency', 'GradCAM', 'GradCAM+', 'Guided Backprop',
    #                 'Integrated Gradients', 'SmoothGrad', 'Square Grad', 'VarGrad', 'Rise',
    #                 'HSIC', 'Sobol', 'Occlusion'],
    #     "faithfulness": [0.18, 0.2, 0.3, 0.32, 0.33, 0.25, 0.26, 0.35, 0.34, 0.38, 0.37, 0.36, 0.29],
    #     "execution_time": [0.5, 1, 2, 3, 4, 10, 20, 50, 100, 150, 80, 120, 200],
    #     "categories": ['White-box', 'White-box', 'Gray-box', 'Gray-box', 'White-box',
    #                    'White-box', 'White-box', 'White-box', 'White-box', 'Black-box',
    #                    'Black-box', 'Black-box', 'Black-box'],
    #     "baselines": [False, False, True, True, False, False, False, False, False, True, True, True, True],
    #     "FORGrad": [True, True, True, True, True, True, True, True, True, False, False, False, False]
    # }
    return data


def plot_explanation_methods(data, show_plot=True):
    """Plots the execution time vs. faithfulness for different explanation methods."""

    # Mapping of categories to colors and markers
    colors = {
        'Black-box': '#377eb8',  # Blue
        'Gray-box': '#4daf4a',  # Green
        'White-box': '#e41a1c',
        "Mask-based": '#377eb8',# Red

    }

    colors = {
        'black': '#000000',
        'orange': '#E69F00',
        'Mask-based': '#56B4E9', #sky_blue
        'green': '#009E73',
        'yellow': '#F0E442',
        'blue': '#0072B2',
        'White-box': '#D55E00', #Brown
        "Gradient-based": '#D55E00',
        'magenta': '#CC79A7'
    }

    markers = {
        'Black-box': 'o',
        "Mask-based": "o",
        'Gray-box': 's',
        'White-box': 's',
        "Gradient-based": 's',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each point by category
    for i in range(len(data['methods'])):
        color = colors[data['categories'][i]]
        marker = markers[data['categories'][i]]
        size = 150  # Marker size

        label = adjust_method_names(data['methods'][i])
        category = data["categories"][i]
        x = data['execution_time'][i]
        y = data['faithfulness'][i]

        ax.scatter(
            x,
            y,
            color=color,
            s=size,
            marker=marker,
            # edgecolor="black",
            linewidth=1.5,  # Thicker edges for more visibility
            label=category,  # Label for all points,
        )

        x_offset, y_offset = 25, 8
        if label == "SquareGrad":
            x_offset, y_offset = 75, 0

        if label == "SmoothGrad":
            x_offset, y_offset = 25, 12

        if label == "Saliency":
            x_offset, y_offset = 30, 8

        elif label == "VarGrad":
            x_offset, y_offset = 55, 0

        elif label in ["ShearletX", "WaveletX"]:
            x_offset, y_offset = 15, -15

        elif label in ["PixelMask"]:
            x_offset, y_offset = 65, 0

        elif label == "Gradient\nInput":
            x_offset, y_offset = 45, 4

        ax.annotate(
            f'{label}', xy=(x, y), xytext=(x_offset, y_offset),  # Text will be offset by 25 units right, 8 units up
            textcoords='offset points',  # Use offset to position text relative to point
            ha='right', fontsize=10,
            fontweight="bold", # Horizontal alignment to 'right'
        )

        # if label in ["WaveletX", "ShearletX", "PixelMask"]:
        #
        #     # Add a horizontal arrow pointing left in the middle of the figure, with the label "AhA"
        #     ax.annotate(
        #         '', xy=(75, y), xytext=(x, y),  # Coordinates for middle positioning
        #         arrowprops=dict(
        #             facecolor=colors["Mask-based"],  # Fill color of the arrow
        #             edgecolor=colors["Mask-based"],  # Border color of the arrow
        #             arrowstyle="->",  # Arrow style
        #             lw=2,  # Linewidth of the arrow
        #         ),
        #         fontsize=18,  # Adjust font size of the annotation
        #         color='black',  # Color of the annotation text
        #         ha='center',
        #         fontweight="bold",  # Horizontal alignment,
        #     )
        #
        #     # Add a horizontal arrow pointing left in the middle of the figure, with the label "AhA"
        #     ax.annotate(
        #         '', xy=(x, y+0.05), xytext=(x, y),  # Coordinates for middle positioning
        #         arrowprops=dict(
        #             facecolor=colors["Mask-based"],  # Fill color of the arrow
        #             edgecolor=colors["Mask-based"],  # Border color of the arrow
        #             arrowstyle="->",  # Arrow style
        #             lw=2,  # Linewidth of the arrow
        #         ),
        #         fontsize=18,  # Adjust font size of the annotation
        #         color='black',  # Color of the annotation text
        #         ha='center',
        #         fontweight="bold",  # Horizontal alignment,
        #     )

    # Add a horizontal arrow pointing left in the middle of the figure, with the label "AhA"
    ax.annotate(
        'StartGrad', xy=(50, 0.35), xytext=(450, 0.35),  # Coordinates for middle positioning
        arrowprops=dict(
            facecolor=colors["Mask-based"],  # Fill color of the arrow
            edgecolor=colors["Mask-based"],  # Border color of the arrow
            arrowstyle="->",  # Arrow style
            lw=2,  # Linewidth of the arrow
        ),
        fontsize=18,  # Adjust font size of the annotation
        color='black',  # Color of the annotation text
        ha='center',
        fontweight="bold", # Horizontal alignment,
    )

    # Add a second arrow orthogonal to the first one (vertical arrow)
    ax.annotate(
        "", xy=(500, 0.43), xytext=(500, 0.37),  # Coordinates for the vertical arrow
        arrowprops=dict(
            facecolor=colors["Mask-based"],  # Fill color of the arrow
            edgecolor=colors["Mask-based"],  # Border color of the arrow
            arrowstyle="->",  # Arrow style
            lw=2,  # Linewidth of the arrow
        )
    )

    # Axis scales and labels
    ax.set_xscale('log')
    # ax.set_ylim([0, 0.5])
    ax.set_xlabel('Execution time (log base 10 scale, seconds)', fontsize=14)
    ax.set_ylabel(r'Faithfulness $\uparrow$', fontsize=14)

    # Increase font size for ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Legend outside of the plot area
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend()

    unique_labels = dict(zip(labels, handles))  # Removes duplicates
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='lower right', fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    # ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    # Customize the legend and make the labels bold
        # Remove grid lines
    # ax.grid(True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend outside the plot
    plt.savefig(
        os.path.join(FIGURES_PATH, f"Performance_Time_Tradeoff_{args.metric}.pdf"), dpi=300, format="pdf"
    )
    if args.show_plot:
        plt.show()


# Main function to execute the plot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the performance speed tradeoff.')
    parser.add_argument('--metric', type=str, help='faithfulness or cp_pixel', default="faithfulness")
    parser.add_argument('--statistics', type=str, help="Sum or mean execution time", default="sum")
    parser.add_argument("--iteration_number", help="statistics of what iteration number to take", default=10)
    parser.add_argument("--show_plot", action="store_true", help="Boolean. If set, it will show all generated plots" )

    args = parser.parse_args()
    args.show_plot = True
    data = get_sample_data(iteration_number=args.iteration_number, statistics=args.statistics)
    plot_explanation_methods(data, args.show_plot)
