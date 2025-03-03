####################################################
# Main file for running the vision experiments
# Author: Buelent Uendes
# Vrije Universiteit Amsterdam
####################################################

import os
import json
import torch
import numpy as np
import wandb
from PIL import Image
import argparse
import pytorch_lightning as pl

from src.vision.waveletX import WaveletX
from src.vision.pixel_RDE import Pixel_RDE
from src.vision.shearletX import ShearletX
from src.vision.saliency_methods import Saliency_Explainer, NoiseGrad

from utils.general.helper import load_yaml_config_file, create_directory
from utils.vision.helper import get_model, preprocess_imagenet_data, \
    get_distortion_various_percentage_levels, get_distorted_random_baselines, \
    get_ssim_for_all_methods
from utils.vision.helper_plotting import plot_explanation_images, plot_distortions, get_mask_histogram
from utils.general.helper_path import CONFIG_PATH, FIGURES_PATH, CUSTOM_PATH, IMAGENET_DATA_PATH, \
    RESULTS_PATH, IMAGENET_TUNING_DATA_PATH, PERFORMANCE_TRADEOFF_PATH
from utils.vision.helper_subset_IMAGENET_creation import create_subset_of_images


def parse_explainers(arg):
    """
    Helper function that returns a list of explainers that are then subsequently run
    :param arg:
    :return:
    """
    if arg == 'all':
        return ['pixelRDE', 'pixelRDE_uniform', 'pixelRDE_p',
                'pixelRDE_saliency', 'pixelRDE_grad_input', 'pixelRDE_smoothgrad', 'pixelRDE_pp',
                'shearletX', 'shearletX_uniform', 'shearletX_p',
                'shearletX_saliency', 'shearletX_smoothgrad', 'shearletX_grad_input', 'shearletX_pp',
                'waveletX', 'waveletX_uniform', 'waveletX_p',
                'waveletX_saliency', 'waveletX_smoothgrad', 'waveletX_grad_input', 'waveletX_pp',
                'IG', 'SG', 'Saliency', 'IG_NG', 'SG_NG', 'Saliency_NG', 'GradCAM', 'LRP']

    else:
        explainers = arg.split(',')
        # Get rid of whitespace
        explainers = [explainer.strip() for explainer in explainers]

        return explainers


def get_random_imagenet_folders(datafolder, number_of_samples=100, same_class=False):
    """
    Pick a random folder from a list of folders
    :param datafolder: List of folders
    :param number_of_samples: int specifying how many random folders you want to select. Default: 100
    :param same_class: bool specifying if all samples need to be from the same class or not
    :return: List of random folder names
    """
    if not same_class:
        folder_idx = np.random.randint(0, len(datafolder), number_of_samples)
        random_folders = [datafolder[idx] for idx in folder_idx]

    else:
        # Otherwise we just get one random class
        folder_idx = list(np.random.randint(0, len(datafolder), 1))
        random_folders = [datafolder[idx] for idx in folder_idx] * number_of_samples

    return random_folders


def get_random_imagenet_samples(folder_path, random_folder, tuning_set_path=IMAGENET_TUNING_DATA_PATH):

    # Idea: I want to exclude the images that are contained in the tuning data set via set difference
    # However, only if we want to use the validation set (when we evaluate the method)
    # if the folder path == tuning set path, then I know that I am hyperparameter tune
    # no restrictions then

    target_path = os.path.join(folder_path, random_folder)

    if folder_path == tuning_set_path: # Then we are tuning over the hyperparameter tuning set,
        allowed_files_to_sample = os.listdir(target_path)

    else:
        # Here I need to restrict the images that we are sampling, as I need to make sure those are not in the
        # tuning set
        tuning_data_path = os.path.join(tuning_set_path, random_folder)
        images_files_target = os.listdir(target_path)
        try:
            images_files_tuning = os.listdir(tuning_data_path)
        except FileNotFoundError:
            images_files_tuning = [] # We create an empty set then

        allowed_files_to_sample = sorted(list(set(images_files_target) - set(images_files_tuning)))

        # Quick check to make sure things worked out as intended
        assert len(allowed_files_to_sample) == (len(images_files_target) - len(images_files_tuning))

    random_idx = np.random.randint(0, len(allowed_files_to_sample))

    return allowed_files_to_sample[random_idx]


def get_all_custom_images(folder_path):
    sample_input = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg'))]
    return sample_input


def parse_input(args):
    """
    Helper function that returns a list of sample input to run the algorithm
    :param args: input argument
    :return: list of images
    """

    if args.input == 'all':
        if args.folder == "Custom":
            sample_input = get_all_custom_images(CUSTOM_PATH)

        elif args.folder == "ImageNet":
            folder_path = IMAGENET_TUNING_DATA_PATH if args.sweep else IMAGENET_DATA_PATH
            folders = os.listdir(folder_path)
            random_folders = get_random_imagenet_folders(folders, args.number_samples, args.same_class)
            random_images = [get_random_imagenet_samples(folder_path, random_folder)
                             for random_folder in random_folders]
            sample_input = [os.path.join(folder, image) for folder, image in zip(random_folders, random_images)]

    else:
        sample_input = args.input.split(',')

    return sample_input


def initialize_empty_list_per_method(methods, random_baselines=True):
    # Instantiate the distortion value dictionary for the results
    distortion_values_list = {method: [] for method in methods}

    if random_baselines:
        distortion_values_list["random baseline"] = []

    return distortion_values_list


def create_specific_save_path(method, seed, model, path):
    # Get the first part to decide which subfolder to store it
    save_dir_name = method.split("_")[0]
    # Create the new directory
    save_path = os.path.join(path, save_dir_name, method, str(seed), model)
    create_directory(save_path)

    return save_path


def check_existence_of_method_in_methods(method_to_check, methods):
    method_to_check = method_to_check.split("_")[0]
    return any(method_to_check in item for item in methods)


def create_specific_figure_save_path(methods, seed, model, path):
    subdirectories = ["cartoonX", "shearletX", "waveletX", "smoothmask", "pixelRDE"]

    for method in subdirectories:
        if check_existence_of_method_in_methods(method, methods):
            save_path = os.path.join(path, method, str(seed), model)
            break
        else:
            save_path = os.path.join(path, "saliency_based", str(seed), model)

    create_directory(save_path)
    return save_path


def save_performance_results(results_path, save_name, results):
    save_path = os.path.join(results_path, save_name + ".json")

    print("saving the results ...")
    with open(save_path, 'w') as file:
        json.dump(results, file)

    print("Done!")


def set_up_device_and_paths(args):
    # Remove non-deterministic behavior of the GPUs, this might slow down everything!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    create_directory(FIGURES_PATH)
    create_directory(PERFORMANCE_TRADEOFF_PATH)
    save_path_figures = create_specific_figure_save_path(args.method, args.seed, args.pretrained_model, FIGURES_PATH)
    folder_path = CUSTOM_PATH if args.folder == "Custom" else IMAGENET_DATA_PATH
    print(f"we save it at  {save_path_figures}")

    return device, save_path_figures, folder_path


def setup_wandb_logging_session(args, hparams_model, method,
                                project_name="startgrad", entity_name="123"):
    wandb.init(
        project=project_name,
        entity=entity_name,
        config=hparams_model,  # Stores the config file of the experiment as a dictionary
        name=args.pretrained_model + "_" + method + "_" + wandb.util.generate_id() + "_" + str(args.seed),
        reinit=True,
        notes="Initial runs for testing wandb"
    )

    args_config_dict = vars(args)

    for key, value in args_config_dict.items():
        if key not in wandb.config.keys():
            wandb.config[key] = value


def get_mask_explainer(args, method, model, hparams_model, device, visualize_single_metrics):
    if method.startswith(('cartoonX', 'waveletX')):
        explainer = WaveletX(model=model, wandb=wandb if args.wandb_logging else None,
                             visualize_single_metrics=visualize_single_metrics,
                             device=device,
                             model_name=args.pretrained_model,
                             **hparams_model)

    elif method.startswith(('pixelRDE', 'smoothmask')):
        explainer = Pixel_RDE(model=model, wandb=wandb if args.wandb_logging else None,
                              visualize_single_metrics=visualize_single_metrics,
                              device=device,
                              model_name=args.pretrained_model,
                              **hparams_model)

    elif method.startswith('shearletX'):
        explainer = ShearletX(model=model, wandb=wandb if args.wandb_logging else None,
                              visualize_single_metrics=visualize_single_metrics,
                              device=device,
                              model_name=args.pretrained_model,
                              **hparams_model)

    elif method.split("_")[-1] == "NG":
        backbone_method = method.split("_")[0]
        explainer = NoiseGrad(model=model, method=backbone_method, device=device)

    else:
        raise ValueError("This mask explainer is not implemented!")

    return explainer


def log_performance_metrics_wandb(performance_metrics, hparams_model):
    number_of_steps = hparams_model['iterations']

    for key, value in performance_metrics.items():
        wandb.log(
            {
                f"{key} results per image": wandb.plot.line_series(
                    xs=np.arange(number_of_steps),
                    ys=[run for run in value],
                    keys=[f"random_image_{i}" for i in range(len(value))],
                    title=f"{key} over time",
                    xname="iteration steps"
                )
            }
        )


def get_distortion_values_for_method(randomize_ascending, method, saliency, sample_input, model, explainer,
                                     hparams_model, device):
    if method.startswith(('cartoonX', 'waveletX', 'shearletX')):
        if isinstance(explainer, WaveletX):
            distortion_values = get_distortion_various_percentage_levels(
                saliency, sample_input, model, explainer, background=hparams_model['perturbation_strategy'],
                randomize_ascending=randomize_ascending, latent_explainer="cartoonX", device=device
            )

        else:
            distortion_values = get_distortion_various_percentage_levels(
                saliency, sample_input, model, explainer, background=hparams_model['perturbation_strategy'],
                randomize_ascending=randomize_ascending, latent_explainer="shearletX", device=device)

    else:
        # This is for gradient-based explanation methods.
        # Background distribution here is following main papers, uniform
        distortion_values = get_distortion_various_percentage_levels(
            saliency, sample_input, model, background="uniform",
            randomize_ascending=randomize_ascending, device=device)

    return distortion_values


def get_distortion_values_random(randomize_ascending, hparams_model, sample_input, model, device):
    # Just in case when we run only saliency-based methods for which we do not have a config file
    background = hparams_model["perturbation_strategy"] if hparams_model else "uniform"
    distortion_values_random = get_distorted_random_baselines(sample_input, model,
                                       background=background,
                                       randomize_ascending=randomize_ascending,
                                       device=device)

    return distortion_values_random


def calculate_average_performance_metrics(
        method, retained_class_probability_per_sample, retained_information_l1_history_per_sample,
        retained_information_pixel_history_per_sample,
        cp_l1_results_per_sample, cp_pixel_results_per_sample,
):

    average_performance_metrics = {
        "retained probability average": list(
            np.mean(np.asarray(retained_class_probability_per_sample[method]), axis=0)),
        "retained information l1 average": list(
            np.mean(np.asarray(retained_information_l1_history_per_sample[method]), axis=0)),
        "retained information pixel average": list(
            np.mean(np.asarray(retained_information_pixel_history_per_sample[method]), axis=0)),
        "cp l1 average": list(np.mean(np.asarray(cp_l1_results_per_sample[method]), axis=0)),
        "cp pixel average": list(np.mean(np.asarray(cp_pixel_results_per_sample[method]), axis=0)),
        "retained probability median": list(
            np.median(np.asarray(retained_class_probability_per_sample[method]), axis=0)),
        "retained probability quantile_025": list(
            np.quantile(np.asarray(retained_class_probability_per_sample[method]), axis=0, q=0.25)),
        "retained probability quantile_075": list(
            np.quantile(np.asarray(retained_class_probability_per_sample[method]), axis=0, q=0.75)),
        "retained probability decentile": list(
            np.quantile(np.asarray(retained_class_probability_per_sample[method]), axis=0, q=0.10)),
        "retained probability one_percentile": list(
            np.quantile(np.asarray(retained_class_probability_per_sample[method]), axis=0, q=0.01)),
        "retained information l1 median": list(
            np.median(np.asarray(retained_information_l1_history_per_sample[method]), axis=0)),
        "retained information pixel median": list(
            np.median(np.asarray(retained_information_pixel_history_per_sample[method]), axis=0)),
        "cp l1 median": list(np.median(np.asarray(cp_l1_results_per_sample[method]), axis=0)),
        "cp l1 p75": list(np.quantile(np.asarray(cp_l1_results_per_sample[method]), axis=0, q=0.75)),
        "cp l1 p25": list(np.quantile(np.asarray(cp_l1_results_per_sample[method]), axis=0, q=0.25)),
        "cp l1 p95": list(np.quantile(np.asarray(cp_l1_results_per_sample[method]), axis=0, q=0.95)),
        "cp pixel median": list(np.median(np.asarray(cp_pixel_results_per_sample[method]), axis=0)),
        "cp pixel p75": list(np.quantile(np.asarray(cp_pixel_results_per_sample[method]), axis=0, q=0.75)),
        "cp pixel p25": list(np.quantile(np.asarray(cp_pixel_results_per_sample[method]), axis=0, q=0.25)),
        "cp pixel p95": list(np.quantile(np.asarray(cp_pixel_results_per_sample[method]), axis=0, q=0.95)),
    }

    return average_performance_metrics


def log_average_performance_metrics_wandb(average_performance_metrics, hparams_model):
    number_of_steps = hparams_model['iterations']
    for key, value in average_performance_metrics.items():
        data = [[x, y] for x, y in zip(np.arange(number_of_steps), value)]
        table = wandb.Table(data=data, columns=["iteration steps", key])
        wandb.log(
            {
                f"{key}": wandb.plot.line(
                    table, "iteration steps", key, title=f"{key} across all images"
                )
            }
        )


def load_and_preprocess_image(folder_path, image, model_name):
    image_path = os.path.join(folder_path, image)
    sample_img = Image.open(image_path).convert('RGB')
    sample_input = preprocess_imagenet_data(sample_img, model_name)
    return sample_input


def get_model_hyperparameter_settings(args, method):
    config_file = 'hparams_' + f'{method}.yaml'
    hparams_model = load_yaml_config_file(os.path.join(CONFIG_PATH, "vision", method.split("_")[0], config_file))

    if args.iterations is not None:
        hparams_model['iterations'] = args.iterations
    if args.lambda_l1 is not None:
        if 'regularization' in hparams_model and 'lambda_l1' in hparams_model['regularization']:
            hparams_model['regularization']['lambda_l1'] = args.lambda_l1
            print("Overriding lambda_l1 parameter to:", args.lambda_l1)
        else:
            print("Warning: lambda_l1 parameter not found in config structure")
            
    if args.lambda_l2 is not None:
        if 'regularization' in hparams_model and 'lambda_l2' in hparams_model['regularization']:
            hparams_model['regularization']['lambda_l2'] = args.lambda_l2
            print("Overriding lambda_l2 parameter to:", args.lambda_l2)
        else:
            print(f"Warning: lambda_l2 parameter not available for {method}")

    return hparams_model


def setup_explainer(args, model, method, device, visualize_single_metrics):

    if method.startswith(('cartoonX', 'pixelRDE', 'shearletX', 'waveletX', 'smoothmask')):
        hparams_model = get_model_hyperparameter_settings(args, method)
        explainer = get_mask_explainer(args, method, model, hparams_model, device, visualize_single_metrics)

    elif method.split("_")[-1] == "NG":
        hparams_model = None
        backbone_method = method.split("_")[0]
        explainer = NoiseGrad(model=model, method=backbone_method, device=device)

    else:
        hparams_model = None
        explainer = Saliency_Explainer(model=model, method=method, device=device,
                                       qtf_standardization=args.qtf_standardization)

    return explainer, hparams_model


def initialize_performance_metrics(methods):
    metrics = [
        "retained_class_probability", "retained_information_pixel", "retained_information_l1",
        "cp_l1_results", "cp_pixel_results",
    ]
    return {metric: {method: [] for method in methods} for metric in metrics}


def update_performance_metrics(performance_metrics_dict, method, explainer):
    performance_metrics_dict["retained_class_probability"][method].append(explainer.retained_class_probability_history)
    performance_metrics_dict["retained_information_pixel"][method].append(explainer.retained_information_pixel_history)
    performance_metrics_dict["retained_information_l1"][method].append(explainer.retained_information_l1_history)
    performance_metrics_dict["cp_l1_results"][method].append(explainer.cp_l1_history)
    performance_metrics_dict["cp_pixel_results"][method].append(explainer.cp_pixel_history)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations',
                        help='Override default config setting for iterations we want to train the algorithm with',
                        default=5,
                        type=int)
    parser.add_argument('--seed', help='seed number', default=123, type=int)
    parser.add_argument('--pretrained_model', type=str,
                        choices=('resnet18', 'vgg16', 'vit_base', 'swin_base', 'swin_t'),
                        help='Please specify which trained model you want to use for running the explanation method',
                        default='resnet18')
    parser.add_argument('--folder', help='Which image folder to use.',
                        choices=('ImageNet', 'Custom'), default='ImageNet')
    parser.add_argument('-x', '--input', help='Which input to explain. Choose "all" if you want to explain all images'
                                              'in the respective folder',
                        type=str, default='all')
    parser.add_argument('--number_samples', help='How many instances you want to explain. '
                                                 'For instance, 100 ImageNet samples',
                        type=int, default=1)
    parser.add_argument('--same_class',
                        help='Boolean indicating if we want to sample from the same class of ImageNet samples. '
                             'Default false',
                        action='store_true')
    parser.add_argument('--method', help='Which explanation method to run '
                                         'Options: cartoonX, shearletX, pixelRDE, IG, SG, GradCAM, LRP'
                                         'pixelRDE_FE, pixelRDE_saliency, Saliency, waveletX or all',
                        type=parse_explainers,  default="waveletX_saliency")
    parser.add_argument("--lambda_l1",
                        help='Override default config setting for lambda 1 we want to train the algorithm with',
                        default=None, type=float)
    parser.add_argument("--lambda_l2",
                        help='Override default config setting for lambda 2 we want to train the algorithm with',
                        default=None, type=float)
    parser.add_argument('--get_distortion_values',
                        help='Boolean. If True, we calculate and save a plot that shows distortion with randomization',
                        action='store_true')
    parser.add_argument('--randomize_ascending',
                        help="Boolean. Randomizes first 10, 20, ... "
                             "percent of the features ascending way, else descending"
                             "Randomize ascending (because we measure distortion) is equivalent to deletion with prob!",
                        action='store_true')
    parser.add_argument("--get_faithfulness", help="If set, we will calculate, both the insertion and deletion score",
                        action="store_true")
    parser.add_argument('--wandb_logging', help='Boolean, if TRUE then wandb logging will be enabled',
                        action='store_true')
    parser.add_argument('--gpu', help='Int. Which gpu to use (if available).', type=int, default=1)
    parser.add_argument('--unnormalized_input', help="Boolean. If set, we do not have a normalization for the input",
                        action="store_false")
    parser.add_argument("--sweep", action="store_true",
                        help="Boolean. If set, a W&B sweep will be executed. "
                             "IMPORTANT: THIS WAS NOT USED FOR PAPER RESULTS.")
    parser.add_argument("--number_sweeps", type=int, help="Number of sweeps to do", default=3)
    parser.add_argument("--add_cp_scores", action="store_true", help="If we want to add cp scores for IG "
                                                                     "and Saliency-based methods")
    parser.add_argument("--qtf_standardization", action="store_true",
                        help="if we want to use qtf_normalization in conjunction with IG methods")
    parser.add_argument("--mask_hist", action="store_true",
                        help="Boolean. If set, mask initializations are stored and plotted.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def setup_sweep(method, number_sweeps):
    """
    Prepares the sweep for one yaml file
    :param method: method to run the sweep for
    :return:
    """
    sweep_config_file = load_yaml_config_file(os.path.join(CONFIG_PATH, "vision", method.split("_")[0], "sweep.yaml"))
    sweep_id = wandb.sweep(sweep=sweep_config_file, project=f"startgrad_sweeps_{method.split('_')[0]}")
    wandb.agent(sweep_id, function=main, count=number_sweeps)


def get_final_insertion_deletion_values(auc_values, randomize_ascending=True):
    mode = "deletion" if randomize_ascending else "insertion"
    final_values = {
        f"{method}_{mode}": np.mean(value) if len(value) > 0 else 0.
        for method, value in auc_values.items()
    }
    return final_values


def calculate_differences(insertions, deletions):
    return [i - d for i, d in zip(insertions, deletions)] if insertions and deletions else []


def get_faithfulness_scores(insertion_scores, deletion_scores):
    faithfulness_scores = {
        method: calculate_differences(insertion_scores[method], deletion_scores[method])
        for method in insertion_scores
    }
    return faithfulness_scores


def main():
    args = parse_arguments()

    device, save_path_figures, folder_path = set_up_device_and_paths(args)

    explanation_maps = {}

    if args.get_distortion_values or args.get_faithfulness:
        distortion_values = initialize_empty_list_per_method(args.method)
        auc_values = initialize_empty_list_per_method(args.method)

        if args.get_faithfulness:
            insertion_scores = initialize_empty_list_per_method(args.method)
            deletion_scores = initialize_empty_list_per_method(args.method)

    if args.mask_hist:
        mask_initializations = initialize_empty_list_per_method(args.method)

    if args.add_cp_scores:
        cp_pixel_scores = initialize_empty_list_per_method(args.method, random_baselines=False)

    sample_inputs = parse_input(args)
    visualize_single_metrics = True if len(sample_inputs) == 1 else False

    performance_metrics_dict = initialize_performance_metrics(args.method)

    subdirectories = ["cartoonX", "shearletX", "waveletX", "smoothmask", "pixelRDE"]

    for method in args.method:

        if check_existence_of_method_in_methods(method, subdirectories):
            # Create save_path for figures. The figure is saved by default in figures/method/seed
            # method refers to the overall method for the latent space, thus pixelRDE, shearlet, wavelet
            save_path_results = create_specific_save_path(method, args.seed, args.pretrained_model, path=RESULTS_PATH)

        model = get_model(args.pretrained_model, device, args.unnormalized_input)
        explainer, hparams_model = setup_explainer(args, model, method, device, visualize_single_metrics)

        if args.wandb_logging:
            setup_wandb_logging_session(args, hparams_model, method)

        for i, image in enumerate(sample_inputs):
            print(f'Method: {method} \nExplaining image {i + 1} / {len(sample_inputs)}')
            sample_input = load_and_preprocess_image(folder_path, image, args.pretrained_model)
            visual_explanation = explainer(sample_input)

            # Get the metrics per method and sample run
            if not isinstance(explainer, Saliency_Explainer):
                update_performance_metrics(performance_metrics_dict, method, explainer)
                saliency = explainer.get_final_mask()

                if args.mask_hist:
                    mask_initializations[method].append(saliency.detach())

            else:
                saliency = visual_explanation

            # We add now for IG and GradCAM the CP-Pixel scores:
            if args.add_cp_scores and isinstance(explainer, Saliency_Explainer):
                cp_pixel_scores[method].append(explainer.cp_pixel_score.item())

            explanation_maps[method] = visual_explanation

            if args.get_distortion_values and not args.get_faithfulness:
                # Important: Getting distortion values for ShearletX method is very computational heavy
                distortion_value = get_distortion_values_for_method(
                    args.randomize_ascending, method, saliency, sample_input, model, explainer, hparams_model, device)
                distortion_value_random = get_distortion_values_random(
                    args.randomize_ascending, hparams_model, sample_input, model, device)
                x_values = np.linspace(0, 1, len(distortion_value))
                # Calculate AUC using the trapezoidal rule
                auc = np.trapz(distortion_value, x_values)
                distortion_values[method].append(distortion_value)
                auc_values[method].append(auc)
                distortion_values["random baseline"].append(distortion_value_random)

            if args.get_faithfulness:
                for truth_value in [1, 0]:
                    distortion_value = get_distortion_values_for_method(
                        truth_value, method, saliency, sample_input, model, explainer, hparams_model,
                        device)
                    x_values = np.linspace(0, 1, len(distortion_value))
                    # Calculate AUC using the trapezoidal rule
                    auc = np.trapz(distortion_value, x_values)
                    insertion_scores[method].append(auc) if truth_value else deletion_scores[method].append(auc)

        if not isinstance(explainer, Saliency_Explainer):
            performance_metrics = {
                "cp-l1": performance_metrics_dict["cp_l1_results"][method],
                "cp-pixel": performance_metrics_dict["cp_pixel_results"][method],
            }

            save_performance_results(save_path_results, "quantitative_performance_measures", performance_metrics)

        if args.add_cp_scores and isinstance(explainer, Saliency_Explainer):
            print(f"The final cp_scores are {cp_pixel_scores}")
            if args.qtf_standardization:
                save_performance_results(RESULTS_PATH, f"quantitative_performance_measures_{args.method}_qtf",
                                     cp_pixel_scores)
            else:
                save_performance_results(RESULTS_PATH, f"quantitative_performance_measures_{args.method}_min_max",
                                     cp_pixel_scores)

        # Now we can get the average results of all performance metrics and plot them at wandb
        if args.wandb_logging and not isinstance(explainer, Saliency_Explainer):
            if not visualize_single_metrics:
                log_performance_metrics_wandb(performance_metrics, hparams_model)

                average_performance_metrics = calculate_average_performance_metrics(
                    method,
                    performance_metrics_dict["retained_class_probability"],
                    performance_metrics_dict["retained_information_l1"],
                    performance_metrics_dict["retained_information_pixel"],
                    performance_metrics_dict["cp_l1_results"],
                    performance_metrics_dict["cp_pixel_results"],
                )

                log_average_performance_metrics_wandb(average_performance_metrics, hparams_model)

            if args.sweep:

                if method.startswith(('pixelRDE', 'smoothmask')):
                    # In this case, cp l1 and cp pixel are the same, thus we need to take only one of those:
                    combined_metric = 1. * average_performance_metrics["cp l1 median"][-1]

                else:
                    combined_metric = 1/2 * average_performance_metrics["cp l1 median"][-1] + \
                                      1/2 * average_performance_metrics["cp pixel median"][-1]

                wandb.log(
                    {
                        "cp l1 median final": average_performance_metrics["cp l1 median"][-1]
                    }
                )
                wandb.log(
                    {
                        "combined metric": combined_metric
                    }
                )

            wandb.finish()

    if args.mask_hist:
        # save the
        get_mask_histogram(mask_initializations, save_path_figures)

    if args.get_faithfulness:
        faithfulness_scores = get_faithfulness_scores(insertion_scores, deletion_scores)
        # Get the average
        summary_faithfulness_scores = {}
        for method, value in faithfulness_scores.items():
            if len(value) > 0 and any(not np.isnan(v) for v in value):
                # Filter out any NaN values
                valid_values = np.array([v for v in value if not np.isnan(v)])
                if len(valid_values) > 0:
                    average_score = np.mean(valid_values)
                    median_score = np.median(valid_values)
                    std_score = np.std(valid_values)
                    std_error_score = std_score / np.sqrt(len(valid_values))
                    #Calculate Interquartile Mean
                    if len(valid_values) >= 3:
                        q25, q75 = np.percentile(valid_values, [25, 75])
                        iqm_mask = (valid_values >= q25) & (valid_values <= q75)
                        iqm_score = np.mean(valid_values[iqm_mask])
                    else:
                        iqm_score = 0.
                else:
                    average_score = median_score = std_score = std_error_score = iqm_score = 0
            else:
                average_score = median_score = std_score = std_error_score = iqm_score = 0
                
            summary_faithfulness_scores[method] = {
                "average": float(average_score),  # Convert to float to ensure JSON serializable
                "median": float(median_score),
                "iqm_score": float(iqm_score),
                "std": float(std_score),
                "std_error": float(std_error_score)
            }

        # Save to JSON file
        with open(os.path.join(PERFORMANCE_TRADEOFF_PATH,
                               f'faithfulness_scores_number_iterations_{args.iterations}.json'), 'w') as json_file:
            json.dump(faithfulness_scores, json_file, indent=4)
        with open(os.path.join(PERFORMANCE_TRADEOFF_PATH,
                               f'summary_faithfulness_scores_number_iterations_{args.iterations}.json'), 'w') as json_file:
            json.dump(summary_faithfulness_scores, json_file, indent=4)

        if args.verbose:
            print(f"Faithfulness scores {faithfulness_scores}")
            print(f"Summary Faithfulness scores {summary_faithfulness_scores}")

    if args.get_distortion_values:
        plot_distortions(distortion_values, save_path_figures, args.folder,
                         randomize_ascending=args.randomize_ascending)

    # Plot the images for each explanation method that we run; only if we did run it for one image!
    if visualize_single_metrics:
        plot_explanation_images(explanation_maps, sample_input, model, save_path_figures,
                                iteration=args.iterations, device=device, seed=args.seed)

        # Check that we can actually do a pairwise-calculation
        if len(args.method) > 1:
            pairwise_ssmi_results = get_ssim_for_all_methods(explanation_maps)
            if args.verbose:
                print(pairwise_ssmi_results)


if __name__ == "__main__":
    args = parse_arguments()
    if args.sweep:
        create_subset_of_images(IMAGENET_DATA_PATH, IMAGENET_TUNING_DATA_PATH)
        if len(args.method) > 1:
            raise ValueError("We can only run sweep for one method instead of 2 or more!")
        setup_sweep(args.method[0], args.number_sweeps)
    else:
        main()

