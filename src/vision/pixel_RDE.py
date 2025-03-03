# Implementation of the Pixel_RDE framework

# Code adapted from:
### Source: https://github.com/KAISER1997/FACTAI_CARTOONX_Reproduce/blob/main/project/pixelRDE.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import Saliency
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ExponentialLR

from utils.general.helper_path import FIGURES_PATH


class Pixel_RDE(torch.nn.Module):

    def __init__(
            self,
            iterations=300,
            learning_rate=1e-1,
            scheduler=None,
            regularization=None,
            sampling_strategy="mask_sampling",
            temperature=0.2,
            sliced_mutual_information="False",
            number_slices=48,
            number_views=64,
            distortion_measure=None,
            model=None,
            batch_size=16,
            mask_init=None,
            perturbation_strategy="gaussian",
            sigma_original=None,
            sigma_distorted=None,
            wandb=None,
            visualize_single_metrics=False,
            grayscale=True,
            device="cpu",
            normalize_gradient=False,
            norm=2.,
            model_name=None,
    ):

        super().__init__()
        self.batch_size = batch_size  # Number of noisy samples to produce
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.regularization = regularization
        self.adaptive_regularization = regularization["adaptive"]
        self.lambda_l1 = regularization["lambda_l1"]
        self.lambda_tv = regularization["lambda_tv"]
        self.delta = regularization["delta"]
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.sliced_mutual_information = sliced_mutual_information
        self.number_slices = number_slices
        self.number_views = number_views
        self.distortion_measure = distortion_measure
        self.mask_init = mask_init
        self.perturbation_strategy = perturbation_strategy
        self.regularization_term = regularization["method"]
        self.epsilon = regularization["epsilon"]
        self.p = regularization["p"]
        self.alpha = regularization["alpha"]
        self.wandb = wandb
        self.visualize_single_metrics = visualize_single_metrics
        self.grayscale = grayscale
        self.device = device
        self.model = model.eval().to(self.device)
        self.sigma_original = sigma_original
        self.sigma_distorted = sigma_distorted
        self.normalize_gradient = normalize_gradient
        self.norm = norm

        # Constants for transforming RGB image to Grayscale image
        self.RED_WEIGHT = 0.299
        self.GREEN_WEIGHT = 0.587
        self.BLUE_WEIGHT = 0.114

        # Std used for standardizing the input for the NN
        # We need the value as we need to rescale the gradient value
        self.STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def __str__(self):
        return "Pixel RDE" if self.lambda_tv == 0. else "Smooth Mask"

    # @time_it(method_name="PixelRDE")
    def __call__(self, x):

        # Check the len of the array
        assert len(x.shape) == 4
        # X has dimensions batch_size, channel, height, width
        x.requires_grad_(False)
        x = x.to(self.device)
        # Get the loss
        self.distortion_loss_history = []
        self.l1_loss_history = []
        self.total_loss_history = []

        self.retained_class_probability_history = []
        self.retained_information_pixel_history = []
        # Important: Pixel and l1 space are the same here!
        self.retained_information_l1_history = []
        self.retained_information_entropy_history = []
        self.retained_information_entropy_no_exp_history = []

        self.cp_l1_history = []
        self.cp_pixel_history = []

        # Get the mean and std deviation for permutation
        x_mean = torch.mean(x)
        x_std = torch.std(x, correction=1)

        # Get the post-softmax probability for argmax predictions
        softmax_prediction_original, self.target_idx = self._get_prediction(x)
        targets_copied = torch.stack(self.batch_size * [softmax_prediction_original])

        # # Initialize the masks
        if self.mask_init["method"] == 'saliency':
            self._get_mask_startgrad(x)

        self.mask = self._get_init_mask(x.to(self.device))

        # # if self.wandb:
        self._get_histogram(self.mask, iteration=0, wandb_logging=self.wandb)

        with torch.no_grad():
            self.total_num_coeff_mask = self.mask.view(1, -1).size(1)

        if self.adaptive_regularization:
            self.lambda_l1_final = self.regularization["lambda_l1"]
            self.lambda_tv_final = self.regularization["lambda_tv"]

        # Initialize optimizer, here Adam
        # We need to give it an iterable to optimize over
        self.optimizer = torch.optim.Adam([self.mask], lr=self.learning_rate)

        if self.scheduler["use"]:
            if self.scheduler["method"] == "cosine":
                self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                                      self.scheduler["T_max"], self.scheduler["eta_min"])
            elif self.scheduler["method"] == "step":
                self.lr_scheduler = StepLR(self.optimizer, step_size=self.scheduler["step_size"],
                                           gamma=self.scheduler["gamma"])

            elif self.scheduler["method"] == "exponential":
                self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.scheduler["gamma"])

        # Logging the results
        print(f"Please wait while the {self.__str__()} model with regularization term {self.regularization_term} "
              f"and mask initialization {self.mask_init} is being trained on gpu {self.device}")

        for iter in tqdm(range(1, self.iterations+1)):

            # Step 1: Sample perturbation
            # Batch_size, gray, height, width
            if self.perturbation_strategy == "gaussian":
                perturbations = x_std * torch.randn(self.batch_size, *x.shape[1:], requires_grad=False,
                                                    dtype=torch.float32, device=self.device) + x_mean

            elif self.perturbation_strategy == "uniform":
                perturbations = torch.rand(self.batch_size, *x.shape[1:], requires_grad=False, dtype=torch.float32,
                                           device=self.device) * (2 * x_std) + (x_mean - x_std)

            else:
                raise ValueError("Please indicate a proper perturbation strategy!")


            x_distorted = self.mask * x + (1 - self.mask) * perturbations

            # if not self.functional_entropy:
            if self.regularization_term == "lp":
                # L1 penalty term
                numerical_constant = 1e-7
                self.sparsity_loss = ((self.mask.abs() + numerical_constant) ** self.p).sum()
                # Regularisation loss
                self.sparsity_loss /= self.total_num_coeff_mask

            elif self.regularization_term == "gaussian_entropy":
                numerical_constant = 1e-7
                squared_coefficients = (torch.abs(self.mask) + numerical_constant) ** 2
                self.sparsity_loss = torch.log(squared_coefficients).sum()
                self.sparsity_loss /= self.total_num_coeff_mask

            elif self.regularization_term == "shannon_entropy":
                numerical_constant = 1e-7
                normalization_term = ((self.mask.abs() + numerical_constant) ** self.p).sum()
                numerator = (self.mask.abs() + numerical_constant) ** self.p
                ratio = numerator / normalization_term

                shannon_entropy = ratio * torch.log(ratio)
                self.sparsity_loss = - shannon_entropy.sum()

            elif self.regularization_term == "renyi_entropy":
                numerical_constant = 1e-7
                normalization_term = ((self.mask.abs() + numerical_constant) ** self.p).sum()
                numerator = (self.mask.abs() + numerical_constant) ** self.p
                ratio = (numerator / normalization_term) ** self.alpha

                self.sparsity_loss = (1 / (1 - self.alpha)) * torch.log(ratio.sum())

            elif self.regularization_term == "log_energy":
                # Important: this is basically the same as the gaussian entropy
                numerical_constant = 1e-7
                self.sparsity_loss = 2 * torch.sum(torch.log(torch.abs(self.mask.view(1, -1)) + numerical_constant))
                self.sparsity_loss /= self.total_num_coeff_mask

            elif self.regularization_term == "log_epsilon":
                normalization = np.log((1 / self.epsilon) + 1)
                self.sparsity_loss = (torch.log((self.mask.abs() / self.epsilon) + 1) / normalization).sum()
                self.sparsity_loss /= self.total_num_coeff_mask

            # Total variance smoothness penalty as introduced in:
            # Real Time Image Saliency for Black Box Classifiers available here: https://arxiv.org/pdf/1705.07857.pdf
            tv_height = (torch.diff(self.mask.squeeze(), n=1, dim=1)**2).sum()
            tv_width = (torch.diff(self.mask.squeeze(), n=1, dim=0) ** 2).sum()
            tv = (tv_height + tv_width) / self.total_num_coeff_mask

            # Step 3: Calculate the distortion
            distorted_softmax_predictions, _ = self._get_prediction(x_distorted, self.target_idx)
            self.distortion_loss = torch.mean((distorted_softmax_predictions.unsqueeze(1) - targets_copied) ** 2, dim=0)

            # Keep track of the history
            self.distortion_loss_history.append(self.distortion_loss.cpu().clone().detach().numpy().item())
            self.l1_loss_history.append(self.sparsity_loss)

            # Update the regularization coefficients in case it is activated:
            if self.adaptive_regularization:
                self._update_lambda(iter)

            # Compute total loss
            self.total_loss = self.distortion_loss + self.lambda_l1 * self.sparsity_loss + self.lambda_tv * tv
            self.total_loss_history.append(self.total_loss.clone().cpu().detach().numpy().item())

            # Calculate the conciseness-preciseness score
            # First retained probability
            retained_class_probability = torch.mean(distorted_softmax_predictions / softmax_prediction_original, dim=0)

            # Second retained image information in the pixel/l1 domain (here pixel and l1 are the same)
            pixel_masked_information = (self.mask.detach() * x).abs().sum().item()
            # pixel_original_information = x.view(1, -1).detach().abs().sum().item()
            pixel_original_information = x.reshape(1, -1).detach().abs().sum().item()

            retained_information_pixel = pixel_masked_information / pixel_original_information
            cp_pixel = retained_class_probability / retained_information_pixel

            # Keep track of the history
            self.cp_l1_history.append(cp_pixel.detach().item())
            self.cp_pixel_history.append(cp_pixel.detach().item())

            self.retained_class_probability_history.append(retained_class_probability.detach().item())
            self.retained_information_pixel_history.append(retained_information_pixel)
            # Important: Pixel and l1 space are the same here!
            self.retained_information_l1_history.append(retained_information_pixel)

            if self.wandb and self.visualize_single_metrics:
                self.wandb.log(
                    {
                        "total_loss": self.total_loss,
                        "distortion_loss": self.distortion_loss,
                        "regularization_term_loss": self.sparsity_loss,
                        "total variance smoothness": tv,
                        "retained_class_probability": retained_class_probability,
                        "retained_information_l1": retained_information_pixel,
                        # IMPORTANT: L1 and pixel space are the same here!
                        "retained_information_pixel": retained_information_pixel,
                        "CP_l1_metric": cp_pixel,  # L1 and pixel are the same here!
                        "CP_pixel_metric": cp_pixel,
                    },
                    step=iter
                )

            self.optimizer.zero_grad()
            self.total_loss.backward()

            if self.normalize_gradient:
                grad_norm = torch.norm(self.mask.grad, p=self.norm)
                self.mask.grad /= (grad_norm + 1e-7)

            self.optimizer.step()

            if self.scheduler["use"]:
                self.lr_scheduler.step()
                print(self.lr_scheduler.get_last_lr())

            # We need to clamp the masks back
            with torch.no_grad():
                self.mask.clamp_(0, 1)

            if self.wandb and self.visualize_single_metrics:
                pixel_x = self.mask * x
                if self.grayscale:
                    pixel_x = (
                            self.RED_WEIGHT * pixel_x[:, 0, :, :] +
                            self.GREEN_WEIGHT * pixel_x[:, 1, :, :] +
                            self.BLUE_WEIGHT * pixel_x[:, 2, :, :]
                    ).unsqueeze(0)

                self.wandb.log({"Images/visual_explanation": self.wandb.Image(pixel_x)}, step=iter)
                self.wandb.log({"Images/mask_explanations": self.wandb.Image(self.mask)}, step=iter)
                # self._get_histogram(self.mask, iteration=iter, save=False, wandb_logging=True)

        # After we have trained the masks, we will return the new image in grayscale
        pixel_x = self.mask * x

        if self.grayscale:
        # Transform to grayscale
            pixel_x = (
                    self.RED_WEIGHT * pixel_x[:, 0, :, :] +
                    self.GREEN_WEIGHT * pixel_x[:, 1, :, :] +
                    self.BLUE_WEIGHT * pixel_x[:, 2, :, :]
            ).unsqueeze(0)

        return pixel_x.squeeze(0)

    # @time_it_QTF(method_name="PixelRDE_startgrad")
    def _get_mask_startgrad(self, x):
        # Enable gradient-based calculation for model weights
        input_saliency = x.clone().detach().requires_grad_(True).to(self.device)
        original_input = x.clone().detach().requires_grad_(False).to(self.device)

        # Important: We do take the gradient with respect to the pre-score! Not the softmax layer!
        # # Own method:
        if self.mask_init["saliency_activation"] == "softmax_layer":
            output_activation = F.softmax(self.model.forward(input_saliency), dim=1)[:, self.target_idx]
            attr = torch.autograd.grad(output_activation, input_saliency, retain_graph=True)[0]

        elif self.mask_init["saliency_activation"] == "output_layer":
            # Using package Captum attribute as this calculate already anyways the gradint w.r.t. to output layer (pre-softmax)
            grad_method = Saliency(self.model.forward)
            attr = grad_method.attribute(input_saliency, target=self.target_idx, abs=False)

        else:
            raise ValueError("Please indicate a proper saliency activation layer!")

        if self.mask_init["noisy_gradients"]:
            gradient_noise = torch.randn_like(attr, dtype=torch.float32, device=self.device)
            attr = attr + gradient_noise

        # Rescale the attribute as we want to get the gradient value w.r.t. to the input of the original NN
        # The standardizer scales the gradient values by 1/std, so we  need to multiply it again
        if self.mask_init["rescaling_gradient"]:
            attr *= self.STD

        if self.mask_init["method"] == "grad_x_input":
            attr *= original_input  # Multiply with the input

        max_attr, _ = torch.max(attr.abs(), dim=1)  # Get the maximum saliency

        # self._get_histogram(max_attr, iteration=0, save=False, wandb_logging=self.wandb)

        mask = self._standardize_saliency_map(max_attr).requires_grad_(True).to(torch.float32)

    def get_final_mask(self):
        return self.mask.squeeze(0)

    def _get_init_mask(self, x):

        if self.mask_init["method"] == "ones":
            # Differentiable mask
            mask = torch.ones(x.size(0), 1, *x.shape[2:], dtype=torch.float32, requires_grad=True, device=self.device)

        elif self.mask_init["method"] == "constant":
            # Differentiable mask
            mask = torch.ones(x.size(0), 1, *x.shape[2:], dtype=torch.float32, requires_grad=True, device=self.device)
            with torch.no_grad():
                mask *= self.mask_init["constant_value"]

        elif self.mask_init["method"] == "zeros":
            mask = torch.zeros(x.size(0), 1, *x.shape[2:], dtype=torch.float32, requires_grad=True, device=self.device)

        elif self.mask_init["method"] == "uniform":
            # Samples uniformly from a mask!
            mask = torch.rand(x.size(0), 1, *x.shape[2:], dtype=torch.float32, requires_grad=True, device=self.device)

        elif self.mask_init["method"] == 'saliency' or self.mask_init["method"] == 'grad_x_input':

            # Enable gradient-based calculation for model weights
            input_saliency = x.clone().detach().requires_grad_(True).to(self.device)
            original_input = x.clone().detach().requires_grad_(False).to(self.device)

            # Important: We do take the gradient with respect to the pre-score! Not the softmax layer!
            # # Own method:
            if self.mask_init["saliency_activation"] == "softmax_layer":
                output_activation = F.softmax(self.model.forward(input_saliency), dim=1)[:, self.target_idx]
                attr = torch.autograd.grad(output_activation, input_saliency, retain_graph=True)[0]

            elif self.mask_init["saliency_activation"] == "output_layer":
            # Using package Captum attribute as this calculate already anyways the gradint w.r.t. to output layer (pre-softmax)
                grad_method = Saliency(self.model.forward)
                attr = grad_method.attribute(input_saliency, target=self.target_idx, abs=False)

            else:
                raise ValueError("Please indicate a proper saliency activation layer!")

            if self.mask_init["noisy_gradients"]:
                gradient_noise = torch.randn_like(attr, dtype=torch.float32, device=self.device)
                attr = attr + gradient_noise

            # Rescale the attribute as we want to get the gradient value w.r.t. to the input of the original NN
            # The standardizer scales the gradient values by 1/std, so we  need to multiply it again
            if self.mask_init["rescaling_gradient"]:
                attr *= self.STD

            if self.mask_init["method"] == "grad_x_input":
                attr *= original_input  # Multiply with the input

            max_attr, _ = torch.max(attr.abs(), dim=1)  # Get the maximum saliency

            # self._get_histogram(max_attr, iteration=0, save=False, wandb_logging=self.wandb)

            mask = self._standardize_saliency_map(max_attr).requires_grad_(True).to(torch.float32)

        elif self.mask_init["method"] == "smoothgrad":

            # We follow the original code implementation as implemented here:
            # https://github.com/PAIR-code/saliency/blob/master/saliency/core/base.py

            # We take the std_deviation_spread to be 0.15 (in fraction of the overall spread max-min), default value 10
            std_spread = 0.15
            n_samples = 250
            input_saliency = x.clone().detach().requires_grad_(True).to(self.device)

            std = std_spread * (torch.max(input_saliency) - torch.min(input_saliency))

            attr_samples = []

            for _ in range(n_samples):
                noise = torch.randn(*x.shape, dtype=torch.float32, device=self.device) * std
                input_saliency_noisy = input_saliency + noise

                if self.mask_init["saliency_activation"] == "softmax_layer":
                    output_activation = F.softmax(self.model.forward(input_saliency_noisy.to(self.device)), dim=1)[:,
                                        self.target_idx]
                    attr = torch.autograd.grad(output_activation, input_saliency_noisy, retain_graph=True)[0]

                elif self.mask_init["saliency_activation"] == "output_layer":
                    # Using package Captum attribute as this calculate already anyways the gradint w.r.t. to output layer (pre-softmax)
                    grad_method = Saliency(self.model.forward)
                    #     # This returns the absolute value of the gradients!
                    attr = grad_method.attribute(input_saliency_noisy, target=self.target_idx)

                else:
                    raise ValueError("Please indicate a proper saliency_activation layer!")

                if self.mask_init["noisy_gradients"]:
                    gradient_noise = torch.randn_like(attr, dtype=torch.float32, device=self.device)
                    attr = attr + gradient_noise

                # Rescale the attribute as we want to get the gradient value w.r.t. to the input of the original NN
                # The standardizer scales the gradient values by 1/std, so we  need to multiply it again
                if self.mask_init["rescaling_gradient"]:
                    attr *= self.STD

                attr_samples.append(torch.abs(attr))

            # Get the average now
            smoothgrad_attr = torch.mean(torch.stack(attr_samples), dim=0)

            max_attr, _ = torch.max(smoothgrad_attr.abs(), dim=1)  # Get the maximum saliency

            mask = self._standardize_saliency_map(max_attr.detach()).requires_grad_(True).to(torch.float32)

        else:
            raise ValueError("Please choose a valid mask initialization option!")

        return mask

    def _standardize_saliency_map(self, saliency_map):
        if self.mask_init["transformation"] == "sqrt":
            if self.mask_init["scaling"] == "min_max":
                saliency_sqrt_min = torch.min(torch.sqrt(saliency_map))
                saliency_sqrt_max = torch.max(torch.sqrt(saliency_map))

                saliency_standardized = (torch.sqrt(saliency_map) - saliency_sqrt_min) / (
                            saliency_sqrt_max - saliency_sqrt_min)

            elif self.mask_init["scaling"] == "sigmoid":
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * (torch.sqrt(saliency_map) - torch.median(torch.sqrt(saliency_map)))
                )

            elif self.mask_init["scaling"] == "sigmoid_iqr":
                iqr = torch.quantile(torch.sqrt(saliency_map), 0.75) - torch.quantile(torch.sqrt(saliency_map), 0.25)
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * ((torch.sqrt(saliency_map) - torch.median(torch.sqrt(saliency_map))) / iqr)
                )

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = torch.sqrt(saliency_map)

            elif self.mask_init["scaling"] == "softmax":
                saliency_map_size = saliency_map.size()
                saliency_softmax_probabilities = F.softmax(torch.sqrt(saliency_map.reshape(1, -1)), dim=1)
                saliency_standardized = saliency_softmax_probabilities.reshape(*saliency_map_size)

            elif self.mask_init["scaling"] == "normalization":
                normalization_term = saliency_map.flatten().sum()
                saliency_standardized = saliency_map / normalization_term

        elif self.mask_init["transformation"] == "exp":
            if self.mask_init["scaling"] == "min_max":
                saliency_exp_min = torch.min(torch.exp(saliency_map) - 1)
                saliency_exp_max = torch.max(torch.exp(saliency_map) - 1)

                saliency_standardized = ((torch.exp(saliency_map) - 1) - saliency_exp_min) / (
                            saliency_exp_max - saliency_exp_min)

        elif self.mask_init["transformation"] == "log":
            if self.mask_init["scaling"] == "min_max":
                saliency_sqrt_min = torch.min(torch.log1p(saliency_map))
                saliency_sqrt_max = torch.max(torch.log1p(saliency_map))

                saliency_standardized = (torch.log1p(saliency_map) - saliency_sqrt_min) / (
                            saliency_sqrt_max - saliency_sqrt_min)

            elif self.mask_init["scaling"] == "sigmoid":
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * (torch.log1p(saliency_map) - torch.median(torch.log1p(saliency_map)))
                )

            elif self.mask_init["scaling"] == "sigmoid_iqr":
                iqr = torch.quantile(torch.log1p(saliency_map), 0.75) - torch.quantile(torch.log1p(saliency_map), 0.25)
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * ((torch.log1p(saliency_map) - torch.median(torch.log1p(saliency_map))) / iqr)
                )

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = torch.log1p(saliency_map)

            elif self.mask_init["scaling"] == "softmax":
                saliency_map_size = saliency_map.size()
                saliency_softmax_probabilities = F.softmax(torch.log1p(saliency_map.reshape(1, -1)), dim=1)
                saliency_standardized = saliency_softmax_probabilities.reshape(*saliency_map_size)

            elif self.mask_init["scaling"] == "normalization":
                normalization_term = saliency_map.flatten().sum()
                saliency_standardized = saliency_map / normalization_term

        elif self.mask_init["transformation"] == "identity":
            # No transformation is applied
            # In case we want to work with the original data (no transformation, yet, highly skewed data)
            if self.mask_init["scaling"] == "min_max":
                saliency_map_min = torch.min(saliency_map)
                saliency_map_max = torch.max(saliency_map)
                saliency_standardized = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            elif self.mask_init["scaling"] == "sigmoid":
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * (saliency_map - torch.median(saliency_map))
                )

            elif self.mask_init["scaling"] == "sigmoid_iqr":
                iqr = torch.quantile(saliency_map, 0.75) - torch.quantile(saliency_map, 0.25)
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * ((saliency_map - torch.median(saliency_map)) / iqr)
                )

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = saliency_map

            elif self.mask_init["scaling"] == "softmax":
                saliency_map_size = saliency_map.size()
                saliency_softmax_probabilities = F.softmax(saliency_map.reshape(1, -1), dim=1)
                saliency_standardized = saliency_softmax_probabilities.reshape(*saliency_map_size)

            elif self.mask_init["scaling"] == "normalization":
                normalization_term = saliency_map.flatten().sum()
                saliency_standardized = saliency_map / normalization_term

        elif self.mask_init["transformation"] == "quantile_transformation":
            print("we use the quantile transformation")
            if len(saliency_map.shape) == 2:
                saliency_map = saliency_map.unsqueeze(0)

            samples = saliency_map.shape[1] * saliency_map.shape[2]
            print(f"Number samples are {samples}")

            quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=samples,
                                                       subsample=None)
            transformed_attr_list = [
                quantile_transformer.fit_transform(mask.reshape(-1, 1).cpu()) for mask in saliency_map
            ]
            saliency_standardized = torch.tensor(np.array(transformed_attr_list), dtype=torch.float32,
                                device=self.device).reshape(*saliency_map.shape).squeeze()

            if self.mask_init["adversarial_gradients"]:
                # So important features will be put to low values and vice versa
                # # Invert the values (0 becomes 1, 1 becomes 0)
                saliency_standardized = 1 - saliency_standardized

            if self.mask_init["shuffle_gradients"]:
                # Shuffle the values randomly so we destroy the whole information
                # Shuffle the values randomly to destroy spatial information
                flat_saliency = saliency_standardized.flatten()
                shuffled_saliency = flat_saliency[torch.randperm(flat_saliency.size(0))]
                saliency_standardized = shuffled_saliency.reshape(saliency_standardized.shape)

        return saliency_standardized.unsqueeze(0)

    def _get_prediction(self, x, predicted_class_idx=None):
        logits_predictions = self.model.forward(x.to(self.device))
        softmax_predictions = F.softmax(logits_predictions, dim=1)

        if predicted_class_idx is None:
            predicted_class_idx = torch.argmax(softmax_predictions).item()
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]
        else:
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]

        return softmax_prediction_top_idx, predicted_class_idx

    def _update_lambda(self, N):
        assert N != 0.
        log_delta_division_factor = np.log(self.delta) / N
        self.lambda_l1 = self.lambda_l1_final * np.exp(log_delta_division_factor)
        self.lambda_tv = self.lambda_tv_final * np.exp(log_delta_division_factor)
        self.lambda_mi = self.lambda_mi_final * np.exp(log_delta_division_factor)

    def _get_histogram(self, *args, iteration=0, save=True, wandb_logging=False):

        plt.figure(figsize=(10, 6))

        for idx, data in enumerate(args, start=0):
            flattened_data = data.view(-1).cpu()
            # Calculate histogram values
            counts, bins, _ = plt.hist(flattened_data.detach().numpy(),
                                       bins=50,
                                       alpha=0.5,
                                       color='#56B4E9')

            # Clear the current histogram
            plt.cla()

            # Replot with scaled frequencies (divide by 100 to show in units of 10^2)
            plt.hist(bins[:-1],
                     bins=bins,
                     weights=counts / 100,
                     alpha=0.5,
                     color='#56B4E9')

        # plt.title(f'Histogram of mask coefficients \n for {self.__str__()}')
        plt.xlabel('Absolute Gradient Value')
        plt.ylabel(r'Frequency [$10^{2}$]')

        if save:
            plt.savefig(os.path.join(FIGURES_PATH,
                                     f"Visualization distribution mask coefficients {self.__str__()} "
                                     f"{self.mask_init['method']} iteration {iteration}.pdf"),
                        dpi=400,
                        format="pdf")

        if wandb_logging:
            self.wandb.log({"Images/mask_coefficients_over_time": self.wandb.Image(plt)}, step=iteration)

        plt.close()



