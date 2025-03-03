# Implementation of the ShearletX algorithm
# Acknowledgments
# Code adjusted from: https://github.com/skmda37/ShearletX/blob/main/code/shearletx.py

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import pyshearlab
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ExponentialLR
from utils.vision.helper import ContrastiveTransformations, contrast_transforms, \
    get_probability_tensor_from_mask, gumbel_softmax, time_it, time_it_QTF
from utils.general.helper_path import FIGURES_PATH

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def torchsheardec2D(torch_X,  torch_shearlets, device="gpu"):
    """Shearlet Decomposition function."""

    Xfreq = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(torch_X))).to(device)
    coeffs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(torch.einsum('bij,bijk->bijk',Xfreq,torch_shearlets)),dim=(-3, 2)))
    return torch.real(coeffs)


def torchshearrec2D(torch_coeffs,  torch_shearlets, dualFrameWeights, dtype=torch.float32, device="gpu"):
    """Shearlet Reconstruction function."""
    torchdualFrameWeights = torch.Tensor(dualFrameWeights).type(dtype).to(device)
    torch_coeffs_freq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch_coeffs)))
    Xfreq = torch.sum(torch_coeffs_freq*torch_shearlets.permute(0, 3, 1, 2),dim=1).to(device)

    return torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Xfreq/torchdualFrameWeights))))


class ShearletX(torch.nn.Module):

    def __init__(self,
                 iterations=300,
                 learning_rate=1e-1,
                 scheduler=None,
                 regularization=None,
                 model=None,
                 batch_size=16,
                 mask_init=None,
                 perturbation_strategy="gaussian",
                 sigma_original=None,
                 sigma_distorted=None,
                 wandb=None,
                 visualize_single_metrics=False,
                 device="gpu",
                 grayscale=True,
                 normalize_gradient=True,
                 norm=2.,
                 model_name="resnet18",
                 ):

        super().__init__()
        self.batch_size = batch_size # Number of noisy samples to produce
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.regularization = regularization
        self.adaptive_regularization = regularization["adaptive"]
        self.lambda_l1 = regularization["lambda_l1"]
        self.lambda_l2 = regularization["lambda_l2"]
        self.lambda_tv = regularization["lambda_tv"]
        self.delta = regularization["delta"]
        self.mask_init = mask_init
        self.perturbation_strategy = perturbation_strategy
        self.regularization_term = regularization["method"]
        self.p = regularization["p"]
        self.alpha = regularization["alpha"]
        self.sigma_original = sigma_original
        self.sigma_distorted = sigma_distorted
        self.wandb = wandb
        self.visualize_single_metrics = visualize_single_metrics
        self.device = device
        self.grayscale = grayscale
        self.normalize_gradient = normalize_gradient
        self.norm = norm
        self.model_name = model_name

        self.model = model.eval().to(self.device)

        # Constants for transforming RGB image to Grayscale image
        self.RED_WEIGHT = 0.299
        self.GREEN_WEIGHT = 0.587
        self.BLUE_WEIGHT = 0.114

        self.STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    # @time_it_QTF(method_name="ShearletX_benchmark")
    def __call__(self, x):

        #Check the len of the array
        assert len(x.shape) == 4
        #X has dimensions batch_size, channel, height, width
        self.x = x.to(self.device)
        self.x.requires_grad_(False)

        # Initialize shearlet system
        if self.model_name == "vit_base" or self.model_name == "swin_base":
            shearletSystem = pyshearlab.SLgetShearletSystem2D(0, self.x.size(-2), self.x.size(-1), 3)
        else:
            shearletSystem = pyshearlab.SLgetShearletSystem2D(0, self.x.size(-2), self.x.size(-1), 4)

        shearlets = shearletSystem['shearlets']
        self.dualFrameWeights = shearletSystem['dualFrameWeights']
        self.torch_shearlets = torch.from_numpy(shearlets[np.newaxis]).type(torch.float32).to(self.device)

        #Initialize lists to store the loss history
        self.distortion_loss_history = []
        self.l1_loss_history = []
        self.spatial_energy_loss_history = []
        self.total_loss_history = []

        self.cp_l1_history = []
        self.cp_pixel_history = []

        self.retained_class_probability_history = []
        self.retained_information_pixel_history = []
        self.retained_information_l1_history = []
        self.retained_information_entropy_history = []
        self.retained_information_entropy_no_exp_history = []

        # Get shearlet coefficients (list of shearlet coefficients per color channels)
        self.shearlet_coeffs = []
        for i in range(x.size(1)):
            coeffs = torchsheardec2D(x[:, i, :, :], self.torch_shearlets, device=self.device).permute(0, 3, 1, 2)
            self.shearlet_coeffs.append(coeffs)

        # Get the grayscale images
        self.x_grayscale = (self.RED_WEIGHT * x[:, 0, :, :] + self.GREEN_WEIGHT * x[:, 1, :, :] + self.BLUE_WEIGHT * x[:, 2, :, :])
        self.x_grayscale.requires_grad_(False)

        self.shearlet_gray = torchsheardec2D(self.x_grayscale, self.torch_shearlets, device=self.device).permute(0, 3, 1, 2)
        assert self.shearlet_gray.size(0) == self.x.size(0)
        assert len(self.shearlet_gray.shape) == 4

        #Get the post-softmax probability for argmax predictions -> this is the probability mask before masking
        softmax_prediction_original, self.target_idx = self._get_prediction(self.x)
        targets_copied = torch.stack(self.batch_size*[softmax_prediction_original])

        # For runtime tracking:
        if self.mask_init["method"] == 'ones':
            self._get_mask_ones()

        if self.mask_init["method"] == "saliency":
            self._get_mask_startgrad()

        # Initialize pixel mask
        self.mask = self._get_init_mask()

        # if self.wandb:
        self._get_histogram(self.mask, iteration=0, save=True, wandb_logging=False)

        # Get the number of mask coefficients
        with torch.no_grad():
            total_num_coeff_mask = self.mask.view(self.mask.size(0), -1).size(-1)

        if self.adaptive_regularization:
            self.lambda_l1_final = self.regularization["lambda_l1"]
            self.lambda_l2_final = self.regularization["lambda_l2"]
            self.lambda_tv_final = self.regularization["lambda_tv"]

        # Initialize the optimizer
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

        print(f"Please wait while the ShearletX model with regularization term {self.regularization_term} "
              f"and mask initialization {self.mask_init} is being trained on gpu {self.device}")

        for iter in tqdm(range(1, self.iterations+1)):
            # Step 1: Sample perturbation
            perturbation = self._get_perturbation()

            # Obfuscate shearlet coefficients
            obf_x = []
            for j in range(x.size(1)):
                obf_shearlet_channel_j = (self.mask.unsqueeze(1) * self.shearlet_coeffs[j].unsqueeze(1) +
                                          (1 - self.mask.unsqueeze(1)) * perturbation).clamp(0, 1).reshape(-1, *self.shearlet_coeffs[j].shape[1:])

                obf_x_channel_j = torchshearrec2D(obf_shearlet_channel_j, self.torch_shearlets, self.dualFrameWeights,
                                                  device=self.device)
                assert tuple(obf_x_channel_j.shape) == (self.batch_size * self.x.size(0), *self.x.shape[-2:]), obf_x_channel_j.shape
                obf_x.append(obf_x_channel_j)
            x_distorted = torch.stack(obf_x, dim=1)
            assert tuple(x_distorted.shape) == (self.batch_size * self.x.size(0), * self.x.shape[1:])

            #Step 2: Calculate the distortion
            distorted_softmax_predictions, _ = self._get_prediction(x_distorted, self.target_idx)
            self.distortion_loss = torch.mean((distorted_softmax_predictions.unsqueeze(1) - targets_copied)**2, dim=0)

            # Get shearletx for spatial regularization
            shearletx = torchshearrec2D(self.mask * self.shearlet_gray,  self.torch_shearlets,
                                        self.dualFrameWeights, device=self.device).clamp(0, 1)

            masked_representation_rgb = [self.mask.detach() * coeffs for coeffs in self.shearlet_coeffs]

            # Compute l1 shearlet norm

            if self.regularization_term == "lp":
                #lp penalty term
                numerical_constant = 1e-7
                self.sparsity_loss = ((self.mask.abs() + numerical_constant) ** self.p).sum()
                self.sparsity_loss /= total_num_coeff_mask

            elif self.regularization_term == "gaussian_entropy":
                # I guess here we get the instability
                numerical_constant = 1e-7
                squared_coefficients = (torch.abs(self.mask) + numerical_constant) ** 2

                self.sparsity_loss = torch.log(squared_coefficients).sum()
                self.sparsity_loss /= total_num_coeff_mask

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

                renyi_entropy = (1 / (1 - self.alpha)) * torch.log(ratio.sum())
                self.sparsity_loss = renyi_entropy.sum()

            elif self.regularization_term == "log_energy":
                numerical_constant = 1e-7
                self.sparsity_loss = 2 * torch.sum(torch.log(torch.abs(self.mask.view(1, -1)) + numerical_constant))
                # Standardize it
                self.sparsity_loss /= total_num_coeff_mask

            elif self.regularization_term == "log_epsilon":
                normalization = np.log((1 / self.epsilon) + 1)
                self.sparsity_loss = (torch.log((self.mask.abs() / self.epsilon) + 1) / normalization).sum()
                self.sparsity_loss /= total_num_coeff_mask

            else:
                raise ValueError("Please use a valid regularization term")

            self.l2_spatial_energy = (shearletx.abs().reshape(shearletx.size(0), -1).sum(dim=-1) /
                                      (np.prod(shearletx.shape[1:]))).sum(dim=-1)


            # Keep track of the history
            self.distortion_loss_history.append(self.distortion_loss.cpu().clone().detach().numpy().item())
            self.l1_loss_history.append(self.sparsity_loss.cpu().detach().numpy().item())
            self.spatial_energy_loss_history.append(self.l2_spatial_energy.cpu().detach().numpy().item())

            # Update the regularization coefficients in case it is activated:
            if self.adaptive_regularization:
                self._update_lambda(iter)

            # Compute total loss
            self.total_loss = self.distortion_loss + self.lambda_l1 * self.sparsity_loss + \
                              self.lambda_l2 * self.l2_spatial_energy
            self.total_loss_history.append(self.total_loss.clone().cpu().detach().numpy().item())

            # Retained probability
            retained_class_probability = torch.mean(distorted_softmax_predictions / softmax_prediction_original, dim=0)

            l1_masked_information = sum([b.abs().sum().item() for b in masked_representation_rgb])
            l1_img_information = sum([a.abs().sum().item() for a in self.shearlet_coeffs])

            # Retained image information
            retained_information_l1 = l1_masked_information / l1_img_information

            # CP measure in the respective domain
            cp_l1 = retained_class_probability / retained_information_l1

            # Pixel domain
            # We just sum up the coefficients of the original image in the pixel domain
            # pixel_original_information = self.x.view(1, -1).detach().abs().sum().item()
            pixel_original_information = self.x.reshape(1, -1).detach().abs().sum().item()

            shearletx_per_channel = [
                torchshearrec2D(self.mask.detach() * coeffs, self.torch_shearlets,
                                self.dualFrameWeights, dtype=torch.float32, device=self.device).clamp(0, 1).unsqueeze(1)
                for coeffs in self.shearlet_coeffs]

            shearletx = torch.cat(shearletx_per_channel, dim=1).clamp(0, 1)
            pixel_masked_information = shearletx.view(1, -1).detach().abs().sum().item()

            retained_information_pixel = pixel_masked_information / pixel_original_information

            cp_pixel = retained_class_probability / retained_information_pixel

            # Keep track of the history
            self.cp_l1_history.append(cp_l1.detach().item())
            self.cp_pixel_history.append(cp_pixel.detach().item())

            self.retained_class_probability_history.append(retained_class_probability.detach().item())
            self.retained_information_pixel_history.append(retained_information_pixel)
            self.retained_information_l1_history.append(retained_information_l1)

            #Logging to wandb
            if self.wandb and self.visualize_single_metrics: #If wandb object exists
                self.wandb.log(
                    {
                        "total_loss": self.total_loss,
                        "distortion_loss": self.distortion_loss,
                        "regularization_term_loss": self.sparsity_loss,
                        "spatial_energy_term_loss": self.l2_spatial_energy,
                        "retained_class_probability": retained_class_probability,
                        "retained_information_l1": retained_information_l1,
                        "retained_information_pixel": retained_information_pixel,
                        "CP_l1_metric": cp_l1,
                        "CP_pixel_metric": cp_pixel
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

            # Project mask into [0,1]
            with torch.no_grad():
                self.mask.clamp_(0, 1)

            if self.wandb and self.visualize_single_metrics:
                if self.grayscale:
                    shearletx_per_channel = [
                        torchshearrec2D(self.mask.detach() * coeffs, self.torch_shearlets, self.dualFrameWeights,
                                        dtype=torch.float32, device=self.device).clamp(0, 1).unsqueeze(1)
                        for coeffs in self.shearlet_gray]

                else:
                    shearletx_per_channel = [
                        torchshearrec2D(self.mask.detach() * coeffs, self.torch_shearlets, self.dualFrameWeights,
                                        dtype=torch.float32, device=self.device).clamp(0, 1).unsqueeze(1)
                        for coeffs in self.shearlet_coeffs]

                shearletx = torch.cat(shearletx_per_channel, dim=1).clamp(0, 1)
                explanation_dict = {"shearletX": shearletx.squeeze(0)}
                # Save the figure for generating the GIFS
                #plot_explanation_images(explanation_dict, self.x, self.model, FIGURES_PATH, device=self.device, iteration=iter)

                self.wandb.log({"Images/visual_explanation": self.wandb.Image(shearletx.detach().squeeze(0))}, step=iter)
                self._get_histogram(self.mask, iteration=iter, save=True, wandb_logging=False)

        if self.grayscale:
            shearletx_per_channel = [
                torchshearrec2D(self.mask.detach() * coeffs, self.torch_shearlets,
                                self.dualFrameWeights, device=self.device).clamp(0, 1).unsqueeze(1)
                for coeffs in self.shearlet_gray]

        else:
            shearletx_per_channel = [
                torchshearrec2D(self.mask.detach() * coeffs, self.torch_shearlets,
                                self.dualFrameWeights, device=self.device).clamp(0, 1).unsqueeze(1)
                for coeffs in self.shearlet_coeffs]

        shearletx = torch.cat(shearletx_per_channel, dim=1).clamp(0, 1)

        return shearletx.squeeze(0)

    def _get_prediction(self, x, predicted_class_idx=None):
        logits_predictions = self.model.forward(x.to(self.device))
        softmax_predictions = F.softmax(logits_predictions, dim=1)

        if predicted_class_idx is None:
            predicted_class_idx = torch.argmax(softmax_predictions).item()
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]
        else:
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]

        return softmax_prediction_top_idx, predicted_class_idx


    # @time_it_QTF(method_name="shearletX_ones")
    def _get_mask_ones(self):
        mask = torch.zeros((self.shearlet_gray.size(0), 1, *self.shearlet_gray.shape[2:]),
                           dtype=torch.float32,
                           device=self.device,
                           requires_grad=True)

        # Just for tracking
    # @time_it_QTF(method_name="shearletX_startgrad")
    def _get_mask_startgrad(self):
        # For tracking
        std_spread = 0.15
        n_samples = 10 if self.mask_init["method"] == "smoothgrad" else 1
        x = self.x.clone().detach().requires_grad_(True).to(self.device)

        std = std_spread * (torch.max(x) - torch.min(x)) \
            if self.mask_init["method"] == "smoothgrad" else 0.

        shearlet_grads = []

        # All we need is to loop over the n_samples we need to get. For smoothgrad we do it n_times, and add noise
        # which we set to 0 in case we do not have smoothgrad method
        for i in range(n_samples):
            sample_shearlet_grad = []

            x_noisy = x.clone().detach().requires_grad_(True).to(self.device)
            # In case of saliency or x_grad we set the std to zero, so there is no noise added. E(X) is 0
            noise = torch.randn_like(self.x, dtype=torch.float32, device=self.device) * std
            x_noisy = x_noisy + noise

            assert x_noisy.requires_grad

            # Get shearlet coefficients (list of shearlet coefficients per color channels)
            shearlet_coeffs = [
                torchsheardec2D(x_noisy[:, i, :, :], self.torch_shearlets,
                                device=self.device).permute(0, 3, 1, 2).requires_grad_(True)
                for i in range(x_noisy.size(1))
            ]

            x_saliency = torch.cat([
                torchshearrec2D(coeffs, self.torch_shearlets,
                                self.dualFrameWeights, device=self.device).unsqueeze(1)
                for coeffs in shearlet_coeffs],
                dim=1)

            # Check that x_saliency also has gradient calculation set to true
            assert x_saliency.requires_grad

            if self.mask_init["saliency_activation"] == "softmax_layer":
                softmax_predictions, _ = self._get_prediction(x_saliency, self.target_idx)
                nll = - torch.log(softmax_predictions)
                nll.backward()

                for s in shearlet_coeffs:

                    # Add noisy gradients, if needed
                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(s.grad, dtype=torch.float32,
                                                          device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        s.grad = s.grad + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        sample_shearlet_grad.append(torch.abs(s.grad * s))

                    else:
                        sample_shearlet_grad.append(torch.abs(s.grad))

            elif self.mask_init["saliency_activation"] == "output_layer":
                output_activation = self.model.forward(x_saliency)[:, self.target_idx]

                for s in shearlet_coeffs:
                    grad = torch.autograd.grad(output_activation, s, retain_graph=True)[0]

                    # Add noisy gradients, if needed
                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(grad, dtype=torch.float32,
                                                          device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        grad = grad + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        sample_shearlet_grad.append(torch.abs(grad * s))

                    else:
                        sample_shearlet_grad.append(torch.abs(grad))

            else:
                raise ValueError(
                    "Please indicate a proper saliency activation option: choices: (softmax_layer, output_layer)."
                )

            max_attr, _ = torch.max(torch.cat(sample_shearlet_grad), dim=0)
            shearlet_grads.append(max_attr.unsqueeze(0))

        # Now we need to average over the samples we have and standardize it
        mask = torch.mean(torch.cat(shearlet_grads), dim=0)
        mask = self._standardize_saliency_map(mask.detach()).requires_grad_(True)

    def _get_init_mask(self):

        if self.mask_init["method"] == 'zeros':
            mask = torch.zeros((self.shearlet_gray.size(0), 1, *self.shearlet_gray.shape[2:]),
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)

        elif self.mask_init["method"] == 'ones':
            mask = torch.ones((self.shearlet_gray.size(0), 1, *self.shearlet_gray.shape[2:]),
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)

        elif self.mask_init["method"] == 'constant':
            mask = torch.ones((self.shearlet_gray.size(0), 1, *self.shearlet_gray.shape[2:]),
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)

            with torch.no_grad():
                mask *= self.mask_init["constant_value"]

        elif self.mask_init["method"] == 'uniform':
            #Creates a random mask initiated between 0 and 1
            mask = torch.rand((self.shearlet_gray.size(0), 1, *self.shearlet_gray.shape[2:]),
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)

        # Results do not change too much if I use saliency-based sampling
        elif self.mask_init["method"] in ["saliency", "grad_x_input", "smoothgrad"]:

            std_spread = 0.15
            n_samples = 10 if self.mask_init["method"] == "smoothgrad" else 1
            x = self.x.clone().detach().requires_grad_(True).to(self.device)

            std = std_spread * (torch.max(x) - torch.min(x)) \
                if self.mask_init["method"] == "smoothgrad" else 0.

            shearlet_grads = []

            # All we need is to loop over the n_samples we need to get. For smoothgrad we do it n_times, and add noise
            # which we set to 0 in case we do not have smoothgrad method
            for i in range(n_samples):
                sample_shearlet_grad = []

                x_noisy = x.clone().detach().requires_grad_(True).to(self.device)
                # In case of saliency or x_grad we set the std to zero, so there is no noise added. E(X) is 0
                noise = torch.randn_like(self.x, dtype=torch.float32, device=self.device) * std
                x_noisy = x_noisy + noise

                assert x_noisy.requires_grad

                # Get shearlet coefficients (list of shearlet coefficients per color channels)
                shearlet_coeffs = [
                    torchsheardec2D(x_noisy[:, i, :, :], self.torch_shearlets,
                                    device=self.device).permute(0, 3, 1, 2).requires_grad_(True)
                    for i in range(x_noisy.size(1))
                ]

                x_saliency = torch.cat([
                    torchshearrec2D(coeffs, self.torch_shearlets,
                                    self.dualFrameWeights, device=self.device).unsqueeze(1)
                    for coeffs in shearlet_coeffs],
                    dim=1)

                # Check that x_saliency also has gradient calculation set to true
                assert x_saliency.requires_grad

                if self.mask_init["saliency_activation"] == "softmax_layer":
                    softmax_predictions, _ = self._get_prediction(x_saliency, self.target_idx)
                    nll = - torch.log(softmax_predictions)
                    nll.backward()

                    for s in shearlet_coeffs:

                        # Add noisy gradients, if needed
                        if self.mask_init["noisy_gradients"]:
                            gradient_noise = torch.randn_like(s.grad, dtype=torch.float32,
                                                              device=self.device) * \
                                             self.mask_init["gradient_noise"]
                            s.grad = s.grad + gradient_noise  # additive noise, for noisy gradient estimation

                        if self.mask_init["method"] == "grad_x_input":
                            sample_shearlet_grad.append(torch.abs(s.grad * s))

                        else:
                            sample_shearlet_grad.append(torch.abs(s.grad))

                elif self.mask_init["saliency_activation"] == "output_layer":
                    output_activation = self.model.forward(x_saliency)[:, self.target_idx]

                    for s in shearlet_coeffs:
                        grad = torch.autograd.grad(output_activation, s, retain_graph=True)[0]

                        # Add noisy gradients, if needed
                        if self.mask_init["noisy_gradients"]:
                            gradient_noise = torch.randn_like(grad, dtype=torch.float32,
                                                              device=self.device) * \
                                             self.mask_init["gradient_noise"]
                            grad = grad + gradient_noise  # additive noise, for noisy gradient estimation

                        if self.mask_init["method"] == "grad_x_input":
                            sample_shearlet_grad.append(torch.abs(grad * s))

                        else:
                            sample_shearlet_grad.append(torch.abs(grad))

                else:
                    raise ValueError(
                        "Please indicate a proper saliency activation option: choices: (softmax_layer, output_layer)."
                    )

                max_attr, _ = torch.max(torch.cat(sample_shearlet_grad), dim=0)
                shearlet_grads.append(max_attr.unsqueeze(0))

            # Now we need to average over the samples we have and standardize it
            mask = torch.mean(torch.cat(shearlet_grads), dim=0)

            mask = self._standardize_saliency_map(mask.detach()).requires_grad_(True)

        else:
            raise ValueError('Need to either pass "zeros", "ones", "random" or "saliency"  '
                             'or "grad_x_input"  or "smoothgrad" for the initialization')

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

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = torch.sqrt(saliency_map)

        elif self.mask_init["transformation"] == "log":
            if self.mask_init["scaling"] == "min_max":
                saliency_log_min = torch.min(torch.log(saliency_map))
                saliency_log_max = torch.max(torch.log(saliency_map))

                saliency_standardized = (torch.log(saliency_map) - saliency_log_min) / (
                            saliency_log_max - saliency_log_min)

            elif self.mask_init["scaling"] == "sigmoid":
                saliency_standardized = torch.sigmoid(
                    self.mask_init["c1"] * (torch.log(saliency_map) - torch.median(torch.log(saliency_map)))
                )

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = torch.log(saliency_map)

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

            elif self.mask_init["scaling"] == "identity":
                saliency_standardized = saliency_map

        elif self.mask_init["transformation"] == "quantile_transformation":
            print("we use the quantile transformation")
            if len(saliency_map.shape) == 2:
                saliency_map = saliency_map.unsqueeze(0)

            samples = saliency_map.shape[1] * saliency_map.shape[2]
            print(f"number_samples shearletX is {samples}")

            # Here we subsample, as otherwise it is very expensive!
            quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=min(10_000, samples),
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

    #@time_it_QTF
    # def _get_transformed_QTF_values(self, quantile_transformer, saliency_map):

    def _get_perturbation(self):
        """
        We only implement for now the standard gaussian permutation with no scaling
        :return:tuple containing the perturbations
        """

        std = torch.std(self.shearlet_gray, dim=[2, 3]).reshape(self.x.size(0), 1, -1, 1, 1)
        mean = torch.mean(self.shearlet_gray, dim=[2, 3]).reshape(self.x.size(0), 1, -1, 1, 1)

        if self.perturbation_strategy == "gaussian":
            #Now we do the parameterization, using the reparameterization trick
            perturbation = std * torch.randn((self.x.size(0), self.batch_size, * self.shearlet_gray.shape[1:]),
                                                   dtype=torch.float32,
                                                   device=self.device,
                                                   requires_grad=False) + mean

        elif self.perturbation_strategy == "uniform":
            perturbation = torch.rand((self.x.size(0), self.batch_size, *self.shearlet_gray.shape[1:]),
                                                   dtype=torch.float32,
                                                   device=self.device,
                                                   requires_grad=False) * (2 * std) + (mean - std)

        return perturbation

    def get_final_mask(self):
        return self.mask.squeeze(0)

    def _update_lambda(self, N):
        assert N != 0.
        log_delta_division_factor = np.log(self.delta) / N
        self.lambda_l1 = self.lambda_l1_final * np.exp(log_delta_division_factor)
        self.lambda_l2 = self.lambda_l2_final * np.exp(log_delta_division_factor)
        self.lambda_tv = self.lambda_tv_final * np.exp(log_delta_division_factor)

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

        # plt.title('Histogram of non-standardized gradient-values per coefficient \nfor ShearletX')
        plt.xlabel('Absolute Gradient Value')
        plt.ylabel(r'Frequency [$10^{2}$]')

        if wandb_logging:
            self.wandb.log({"Images/mask_coefficients_over_time": self.wandb.Image(plt)}, step=iteration)

        if save:
            plt.savefig(os.path.join(FIGURES_PATH, "Visualization distribution mask coefficients shearletx.pdf"),
                        dpi=400,
                        format="pdf")
        plt.close()
