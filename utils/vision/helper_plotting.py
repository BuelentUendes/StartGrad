import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from utils.general.helper_path import IMAGENET_PATH, FIGURES_PATH

#Define ggplot2 style
# plt.style.use('ggplot')

def get_imagenet_labels() -> List[str]:
    """
    Gets the imagenet labels that are stored in a txt.file
    :return: List of the labels of the predictions
    """
    with open(os.path.join(IMAGENET_PATH, 'imagenet_classes.txt'), 'r') as f:
        categories = [s.strip() for s in f.readlines()] #strip() removes leading and trailing whitespaces
    return categories


def plot_original_explanation_img(original_img, explanation_img, explainer: str, save_loc):

    #Show the final explanation of the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img[0].cpu().detach().permute(1, 2, 0))
    # plt.title('Original image', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.imshow(explanation_img.cpu().detach().permute(1, 2, 0), cmap='gray')
    plt.title(f'{explainer} explanation')
    plt.savefig(os.path.join(save_loc, f'{explainer}' + '.png'), dpi=400, format='png')
    plt.show()
    plt.close()


def plot_explanation_images(explanations, original_img, model, save_loc, wandb=None, device="gpu",
                            iteration=None, seed=123):

    #Get the imagenet_labels for plotting the predicted label
    imagenet_labels = get_imagenet_labels()
    label = torch.argmax(model.forward(original_img.to(device)), dim=1).cpu()
    predicted_label = imagenet_labels[label]

    plt.figure(figsize=(12, 6))
    # plot the original image
    plt.subplot(1, len(explanations)+1, 1)
    plt.imshow(original_img[0].cpu().detach().permute(1, 2, 0))
    plt.title(f'Model prediction: {predicted_label}', fontsize=12)
    plt.axis('off')

    method_dict = {
        "pixelRDE_saliency": "PixelMask (StartGrad)",
        "pixelRDE_uniform": "PixelMask (Uniform)",
        "pixelRDE": "PixelMask (All-ones)",
        "waveletX_saliency": "WaveletX (StartGrad)",
        "waveletX_uniform": "WaveletX (Uniform)",
        "waveletX": "WaveletX (All-ones)",
        "shearletX_saliency": "ShearletX (StartGrad)",
        "shearletX_uniform": "ShearletX (Uniform)",
        "shearletX": "ShearletX (All-ones)",

    }

    for i, (name, explanation) in enumerate(explanations.items()):

        plt.subplot(1, len(explanations)+1, i+2)
        if isinstance(explanation, torch.Tensor):
            plt.imshow(explanation.cpu().detach().permute(1, 2, 0), cmap='copper') #'gray'
        else:
            plt.imshow(explanation[0].cpu().detach().permute(1, 2, 0), cmap='copper')

        plt.title(f'{method_dict[name]}', fontsize=12)
        # if iteration is not None:
        #     plt.title(f'Explanation method: {name} \n iteration: {iteration}', fontsize=12)
        # else:
        #     plt.title(f'Explanation method: \n {name}', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    if iteration is not None:
        plt.savefig(os.path.join(save_loc, f'Comparison_explainers_{name}_{iteration}_{seed}.png'), dpi=400, format='png')
    else:
        plt.savefig(os.path.join(save_loc, f'Comparison_explainers_{name}_{seed}.png'), dpi=400, format='png')
    if wandb:
        image = wandb.Image(plt)
        wandb.log({'figures/explanation_image': image})
    plt.show()
    plt.close()


def plot_loss_training(dict_of_losses, save_loc, step_size=50):

    plt.figure(figsize=(12, 6))

    assert type(step_size) is int, 'step_size needs to be an integer!'

    #Get the average of all training samples
    average_total_loss = {
        key: np.mean(value, axis=0) for key, value in dict_of_losses.items()
    }
    for method, values in average_total_loss.items():
        indices = np.arange(0, len(values), step_size)
        plt.plot(indices, values[::step_size], label=method)

    if list(average_total_loss.keys())[0].startswith('cartoon'):
        mask_explainer = 'cartoonX'

    elif list(average_total_loss.keys())[0].startswith('pixelRDE'):
        mask_explainer = 'pixelRDE'

    else:
        mask_explainer = ''

    plt.xlabel('Training iterations')
    plt.xticks(indices)
    plt.ylabel('Total loss')
    plt.title('Loss over different compression loss terms')
    plt.legend()
    plt.savefig(os.path.join(save_loc, f'total_loss_over_time_{mask_explainer}.png'), dpi=400, format='png')
    plt.show()
    plt.close()


def plot_distortions(distortion_values, save_loc, dataset, percentage_list=None,
                     randomize_ascending=False, plot_RDE=False, l1_lambda=2.):
    plt.figure(figsize=(10, 6))
    if percentage_list is None:
        percentage_list = [round(x, 2) for x in np.arange(0.0, 105, 5)]

    for method, values in distortion_values.items():
        if plot_RDE:
            if method == "random baseline":
                continue

            if method.endswith("uniform"):
                values = [[value + 0.5 * l1_lambda for value in value_list] for value_list in values]
                method = "uniform initialization $\lambda = 2$"

            elif method.endswith("saliency"):
                values = [[value + 0.5 * l1_lambda for value in value_list] for value_list in values]
                method = "StartGrad (ours) $\lambda = 2$"

            else:
                method = "all-ones initialization $\lambda = 2$"
                values = [[value + 1. * l1_lambda for value in value_list] for value_list in values]

            #Take the mean for each percentage level
            average_values = np.asarray(values).mean(axis=0)
            plt.plot(percentage_list, average_values, label=method)

        else:
            # Take the mean for each percentage level
            average_values = np.asarray(values).mean(axis=0)
            plt.plot(percentage_list, average_values, label=method)

    # plt.title("Distortion values per method")
    plt.ylabel("RDE loss", fontsize=14) if plot_RDE else plt.ylabel("L2 distortion", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if randomize_ascending:
        plt.xlabel("Randomized relevant components [in %]", fontsize=14)
        save_name = f"Distortion_values_randomized_{dataset}.pdf"
    else:
        plt.xlabel("Non-randomized relevant components [in %]", fontsize=14)
        save_name = f"Distortion_values_non_randomized_{dataset}.pdf"
    plt.legend(loc="lower left", fontsize=14)
    plt.grid(False)
    plt.savefig(os.path.join(save_loc, save_name), dpi=400, format="pdf")
    plt.show()
    plt.close()


def get_mask_histogram(saliency_values_dict, save_path):

    for method, masks in saliency_values_dict.items():
        if len(masks) > 0: #Exclude empty lists
            plt.figure(figsize=(6, 4))
            saliency_values = torch.stack(masks)
            number_samples = saliency_values.shape[0]
            values = saliency_values.flatten().cpu().numpy()
            plt.hist(values, bins=500,
                     alpha=0.8,  weights=np.ones_like(values) / 10e2)

            plt.xlabel('Absolute gradient value', fontsize=8)
            # plt.yscale('log')
            plt.ylabel('Frequency in $[10^{2}]$', fontsize=8)
            plt.xlim(0, 1)  # Adjust the range as needed

            # Adjust tick label font sizes
            plt.xticks(fontsize=8)  # Smaller fontsize for x-axis ticks
            plt.yticks(fontsize=8)

            # # Set y-axis ticks at intervals of 10^6
            # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0e}'.format(x)))

            plt.grid(True)
            plt.savefig(os.path.join(save_path,
                                     f"Visualization distribution gradient distribution "
                                     f"{method} samples {number_samples}.png"), dpi=400,
                        format="png")
            plt.close()