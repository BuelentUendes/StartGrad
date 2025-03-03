# Code adapted from:
# https://github.com/KAISER1997/FACTAI_CARTOONX_Reproduce/blob/main/project/cartoonX.py
import numpy as np
import os
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ExponentialLR
from sklearn.preprocessing import QuantileTransformer

from utils.vision.helper import ContrastiveTransformations, contrast_transforms, \
    get_probability_tensor_from_mask, gumbel_softmax, time_it, time_it_QTF

# BASE_DATA_PATH = os.getcwd()
# DATA_PATH = os.path.join(BASE_DATA_PATH, './../', 'data', 'ImageNet')
#
# # Standard measures for mean and std for the transformation for IMAGENET model_architecture
# NORM_MEAN = np.array([0.485, 0.456, 0.406])
# NORM_STD = np.array([0.229, 0.224, 0.225])


class WaveletX(torch.nn.Module):

    def __init__(self,
                 iterations=300,
                 learning_rate=1e-1,
                 scheduler=None,
                 regularization=None,
                 distortion_measure=None,
                 model=None,
                 wave='db3',
                 mode='zero',
                 J=5,
                 batch_size=16,
                 mask_init=None,
                 perturbation_strategy="gaussian",
                 alpha=None,
                 compression="pixel",
                 sigma_original=None,
                 sigma_distorted=None,
                 wandb=None,
                 visualize_single_metrics=False,
                 grayscale=True,
                 device="gpu",
                 normalize_gradient=False,
                 norm=2.,
                 model_name="resnet18",
                 ):

        super().__init__()
        self.batch_size = batch_size  # Number of noisy samples to produce
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.regularization = regularization
        self.adaptive_regularization = regularization["adaptive"]
        self.lambda_l1 = regularization["lambda_l1"]
        self.lambda_l2 = regularization["lambda_l2"]
        self.lambda_tv = regularization["lambda_tv"]
        self.delta = regularization["delta"]
        self.wave = wave
        self.mode = mode
        self.J = J
        self.distortion_measure = distortion_measure
        self.mask_init = mask_init
        self.perturbation_strategy = perturbation_strategy
        self.regularization_term = regularization["method"]
        self.p = regularization["p"]
        self.epsilon = regularization["epsilon"]
        self.alpha = regularization["alpha"]
        self.compression = compression
        self.alpha = alpha
        self.sigma_original = sigma_original
        self.sigma_distorted = sigma_distorted
        self.wandb = wandb
        self.visualize_single_metrics = visualize_single_metrics
        self.grayscale = grayscale
        self.normalize_gradient = normalize_gradient
        self.norm = norm
        self.device = device
        self.model = model.eval().to(self.device)

        # Constants for transforming RGB image to Grayscale image
        self.RED_WEIGHT = 0.299
        self.GREEN_WEIGHT = 0.587
        self.BLUE_WEIGHT = 0.114

        # Initialize the forward DWT and backward DWT
        self.forward_dwt = DWTForward(J=J, wave=wave, mode=mode).to(self.device)
        self.inverse_dwt = DWTInverse(mode=mode, wave=wave).to(self.device)

        self.STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    # Name for printing statement
    def __str__(self):
        name = 'cartoonX' if self.lambda_l2 == 0. else 'waveletX'
        return name

    # @time_it_QTF(method_name="WaveletX_benchmark")
    def __call__(self, x):

        # Check the len of the array
        assert len(x.shape) == 4
        # X has dimensions batch_size, channel, height, width
        self.x = x.to(self.device)
        self.x.requires_grad_(False)

        # Initialize lists to store the loss history
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

        # Get the grayscale images
        x_grayscale = (
                self.RED_WEIGHT * x[:, 0, :, :] + self.GREEN_WEIGHT * x[:, 1, :, :]
                + self.BLUE_WEIGHT * x[:, 2, :, :]
        ).unsqueeze(0)

        x_grayscale.requires_grad_(False)

        # Get the forward DWT
        self.yl, self.yh = self.forward_dwt(x.to(self.device))

        # We need those grayscale images later, as we want to represent the smooth-like explanations in grayscale
        # Dimension low scale
        # batch_size, channel, scale, scale
        # Dimensions detailed approximations
        # batch_size, channel, 3, scale, scale (3 -> refers to horizontal, vertical, diagonal)
        yl_gray, yh_gray = self.forward_dwt(x_grayscale.to(self.device))

        # Get the post-softmax probability for argmax predictions -> this is the probability mask before masking
        softmax_prediction_original, self.target_idx = self._get_prediction(x)
        targets_copied = torch.stack(self.batch_size * [softmax_prediction_original])

        # For tracking purposes of the runtime for StartGrad
        # For runtime tracking:
        if self.mask_init["method"] == "saliency":
            self._get_mask_startgrad()

        # Initialize the masks
        self.mask_yl, self.mask_yh = self._get_init_mask(self.yl, self.yh)

        with torch.no_grad():
            num_coeff_mask_yl = self.mask_yl.view(1, -1).size(1)
            num_coeff_mask_yh = np.sum([s.view(1, -1).size(1) for s in self.mask_yh])
            total_num_coeff_mask = num_coeff_mask_yl + num_coeff_mask_yh

        if self.adaptive_regularization:
            self.lambda_l1_final = self.regularization["lambda_l1"]
            self.lambda_l2_final = self.regularization["lambda_l2"]
            self.lambda_tv_final = self.regularization["lambda_tv"]

        # Initialize optimizer, here Adam
        # We need to give it an iterable to optimize over
        self.optimizer = torch.optim.Adam([self.mask_yl] + self.mask_yh, lr=self.learning_rate)

        if self.scheduler["use"]:
            if self.scheduler["method"] == "cosine":
                self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                                      self.scheduler["T_max"], self.scheduler["eta_min"])
            elif self.scheduler["method"] == "step":
                self.lr_scheduler = StepLR(self.optimizer, step_size=self.scheduler["step_size"],
                                           gamma=self.scheduler["gamma"])

            elif self.scheduler["method"] == "exponential":
                self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.scheduler["gamma"])

        print(f"Please wait while the {self.__str__()} model with regularization term {self.regularization_term}, "
              f"compression space {self.compression},"
              f" mask initialization {self.mask_init}"
              f" is being trained on gpu {self.device}")

        for iter in tqdm(range(1, self.iterations+1)):
            # Step 1: Sample perturbation
            perturbation_yl, perturbation_yh = self._get_perturbation(self.yl, self.yh)

            # Step 2: Calculate the distorted images with the mask and the perturbation
            yl_distorted = self.mask_yl * self.yl + (1 - self.mask_yl) * perturbation_yl
            yh_distorted = [self.mask_yh[i] * self.yh[i] + (1 - self.mask_yh[i]) * perturbation_yh[i] for i in
                            range(len(self.yh))]

            # Step 3: Inverse DWT and clamp back to (0,1)
            x_distorted = self.inverse_dwt((yl_distorted, yh_distorted)).clamp(0, 1).to(self.device)

            # if not self.functional_entropy:
            if self.regularization_term == "lp":
                numerical_constant = 1e-7
                # # Lp penalty term
                self.sparsity_loss = ((self.mask_yl.abs() + numerical_constant) ** self.p).sum()

                for mask in self.mask_yh:
                    self.sparsity_loss += ((mask.abs() + numerical_constant) ** self.p).sum()

                # Regularisation loss
                self.sparsity_loss /= total_num_coeff_mask

            elif self.regularization_term == "gaussian_entropy":
                numerical_constant = 1e-7
                self.sparsity_loss = torch.log(torch.abs(self.mask_yl + numerical_constant) ** 2).sum()

                for mask in self.mask_yh:
                    squared_coefficients = (torch.abs(mask) + numerical_constant) ** 2
                    gaussian_entropy = torch.log(squared_coefficients).sum()
                    self.sparsity_loss += gaussian_entropy
                    self.sparsity_loss /= total_num_coeff_mask

            elif self.regularization_term == "shannon_entropy":
                numerical_constant = 1e-7
                normalization_term = ((self.mask_yl.abs() + numerical_constant) ** self.p).sum()
                numerator = (self.mask_yl.abs() + numerical_constant) ** self.p
                ratio = numerator / normalization_term

                shannon_entropy = ratio * torch.log(ratio)
                self.sparsity_loss = - shannon_entropy.sum()

                for mask in self.mask_yh:
                    normalization_term = ((mask.abs() + numerical_constant) ** self.p).sum()
                    numerator = (mask.abs() + numerical_constant) ** self.p
                    ratio = numerator / normalization_term

                    shannon_entropy = ratio * torch.log(ratio)
                    self.l1_loss -= shannon_entropy.sum()

            elif self.regularization_term == "renyi_entropy":
                numerical_constant = 1e-7
                normalization_term = ((self.mask_yl.abs() + numerical_constant) ** self.p).sum()
                numerator = (self.mask_yl.abs() + numerical_constant) ** self.p
                ratio = (numerator / normalization_term) ** self.alpha

                self.sparsity_loss = (1 / (1 - self.alpha)) * torch.log(ratio.sum())

                for mask in self.mask_yh:
                    normalization_term = ((mask.abs() + numerical_constant) ** self.p).sum()
                    numerator = (mask.abs() + numerical_constant) ** self.p
                    ratio = (numerator / normalization_term) ** self.alpha

                    self.sparsity_loss += (1 / (1 - self.alpha)) * torch.log(ratio.sum())

            elif self.regularization_term == "log_energy":
                numerical_constant = 1e-7
                self.sparsity_loss = 2 * torch.sum(torch.log(torch.abs(self.mask_yl.view(1, -1)) + numerical_constant))

                for mask in self.mask_yh:
                    self.sparsity_loss += 2 * torch.sum(torch.log(torch.abs(mask.view(1, -1)) + numerical_constant))

                self.sparsity_loss /= total_num_coeff_mask

            elif self.regularization_term == "log_epsilon":
                normalization = np.log((1 / self.epsilon) + 1)
                self.sparsity_loss = (torch.log((self.mask_yl.abs() / self.epsilon) + 1) / normalization).sum()

                for mask in self.mask_yh:
                    self.sparsity_loss += (torch.log((mask.abs() / self.epsilon) + 1) / normalization).sum()

                self.sparsity_loss /= total_num_coeff_mask


            if self.compression == "pixel":
                spatial_energy = self.inverse_dwt(
                    (self.mask_yl * self.yl, [mask * yh for mask, yh in zip(self.mask_yh, self.yh)])
                ).clamp(0, 1)

                self.l2_spatial_energy = spatial_energy.abs().reshape(spatial_energy.size(0), -1).sum(dim=-1)
                number_coefficients_spatial = np.prod(spatial_energy.shape[1:])
                self.l2_spatial_energy /= number_coefficients_spatial

            elif self.compression == "l1":
                compressed_yl = (self.mask_yl * self.yl).abs().reshape(1, -1).sum(dim=-1)
                compressed_yh = 0
                for mask, yh in zip(self.mask_yh, self.yh):
                    compressed_yh += (mask * yh).abs().reshape(1, -1).sum(dim=-1)
                self.l2_spatial_energy = (compressed_yl + compressed_yh) / total_num_coeff_mask

            elif self.compression == "distortion":
                spatial_energy = x_distorted.clone().mean(dim=0).unsqueeze(0)
                self.l2_spatial_energy = spatial_energy.abs().reshape(spatial_energy.size(0), -1).sum(dim=-1)
                number_coefficients_spatial = np.prod(spatial_energy.shape[1:])
                self.l2_spatial_energy /= number_coefficients_spatial

            elif self.compression == "mask_information":
                # Measures mask information as discussed in "Explaining Time Series Predictions with Dynamic Masks"
                # https://arxiv.org/pdf/2106.05303.pdf
                numerical_constant = 1e-7
                mask_information_yl = - torch.log(self.mask_yl.flatten().clamp(0.+numerical_constant, 1.-numerical_constant)).sum()
                mask_information_yh = 0
                for mask in self.mask_yh:
                    mask_information_yl -= torch.log(mask.flatten().clamp(0.+numerical_constant, 1.-numerical_constant)).sum()

                # We minimize the function in PyTorch
                self.l2_spatial_energy = - (mask_information_yl + mask_information_yh)
                # Normalize it
                self.l2_spatial_energy /= total_num_coeff_mask

            # Step 4: Calculate the distortion
            distorted_softmax_predictions, _ = self._get_prediction(x_distorted, self.target_idx)
            self.distortion_loss = torch.mean((distorted_softmax_predictions.unsqueeze(1) - targets_copied) ** 2, dim=0)

            # Keep track of the history
            self.distortion_loss_history.append(self.distortion_loss.cpu().clone().detach().numpy().item())
            self.l1_loss_history.append(self.sparsity_loss.cpu().detach().numpy().item())
            self.spatial_energy_loss_history.append(self.l2_spatial_energy.cpu().detach().numpy().item())

            # Update the regularization coefficients in case it is activated:
            if self.adaptive_regularization:
                self._update_lambda(iter)

            # Compute total loss
            self.total_loss = self.distortion_loss + self.lambda_l1 * self.sparsity_loss +\
                              self.lambda_l2 * self.l2_spatial_energy
            self.total_loss_history.append(self.total_loss.clone().cpu().detach().numpy().item())

            # Calculate the conciseness-preciseness score (CP)
            # Code adapted from: https://github.com/skmda37/ShearletX/blob/main/code/waveletx.py

            # First: Retained probability here we take the mean as we have batched it (softmax_prediction_original is a tensor)
            retained_class_probability = torch.mean(distorted_softmax_predictions / softmax_prediction_original, dim=0)

            # Second: Retained image information
            # Relative sparsity
            l1_masked_information = (self.mask_yl.detach() * self.yl).abs().sum().item() + sum(
                [(mask.detach() * yh).abs().sum().item() for mask, yh in zip(self.mask_yh, self.yh)])
            l1_original_information = self.yl.abs().sum().item() + sum([y.detach().abs().sum().item() for y in self.yh])

            retained_information_l1 = l1_masked_information / l1_original_information

            # CP measure (in the respective domain) as defined in the paper
            cp_l1 = retained_class_probability / retained_information_l1

            # CP measure in the pixel domain
            # Check this!
            # We just sum up the coefficients of the original image in the pixel domain
            # pixel_original_information = self.x.view(1, -1).detach().abs().sum().item()
            pixel_original_information = self.x.reshape(1, -1).detach().abs().sum().item()
            yl = self.mask_yl * self.yl
            yh = [self.mask_yh[i] * self.yh[i] for i in range(len(self.yh))]
            masked_img = self.inverse_dwt((yl, yh)).clamp(0, 1).to(self.device).cpu()
            pixel_masked_information = masked_img.view(1, -1).detach().abs().sum().item()

            retained_information_pixel = pixel_masked_information / pixel_original_information

            cp_pixel = retained_class_probability / retained_information_pixel

            # Keep track of the history
            self.cp_l1_history.append(cp_l1.detach().item())
            self.cp_pixel_history.append(cp_pixel.detach().item())

            self.retained_class_probability_history.append(retained_class_probability.detach().item())
            self.retained_information_pixel_history.append(retained_information_pixel)
            self.retained_information_l1_history.append(retained_information_l1)

            # Logging to wandb
            if self.wandb and self.visualize_single_metrics:  # If wandb object exists
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
                grad_norm_yl = (torch.norm(self.mask_yl, p=self.norm) + 1e-7)
                self.mask_yl.grad /= grad_norm_yl

                for mask in self.mask_yh:
                    mask.grad /= (torch.norm(mask.grad, p=self.norm) + 1e-7)

            self.optimizer.step()

            if self.scheduler["use"]:
                self.lr_scheduler.step()

            # We need to clamp the masks back
            with torch.no_grad():
                self.mask_yl.clamp_(0, 1)
                for i in range(len(self.mask_yh)):
                    self.mask_yh[i].clamp_(0, 1)

            if self.wandb and self.visualize_single_metrics:
                # After we have trained the masks, we will return the new image in grayscale
                if self.grayscale:
                    sparse_mask_yl = self.mask_yl * yl_gray
                    sparse_mask_yh = [mask * coeff for mask, coeff in zip(self.mask_yh, yh_gray)]

                else:
                    sparse_mask_yl = self.mask_yl * self.yl
                    sparse_mask_yh = [mask * coeff for mask, coeff in zip(self.mask_yh, self.yh)]

                cartoonX = self.inverse_dwt((sparse_mask_yl, sparse_mask_yh)).cpu().squeeze(0)
                self.wandb.log({"Images/visual_explanation": self.wandb.Image(cartoonX)}, step=iter)

        # After we have trained the masks, we will return the new image in grayscale/RGB
        if self.grayscale:
            sparse_mask_yl = self.mask_yl * yl_gray
            sparse_mask_yh = [mask * coeff for mask, coeff in zip(self.mask_yh, yh_gray)]

        else:
            sparse_mask_yl = self.mask_yl * self.yl
            sparse_mask_yh = [mask * coeff for mask, coeff in zip(self.mask_yh, self.yh)]

        cartoonX = self.inverse_dwt((sparse_mask_yl, sparse_mask_yh)).cpu().squeeze(0)

        return cartoonX

    # @time_it_QTF(method_name="waveletX_startgrad")
    def _get_mask_startgrad(self):
        std_spread = 0.15
        n_samples = 10 if self.mask_init["method"] == "smoothgrad" else 1
        x = self.x.clone().detach().requires_grad_(True).to(self.device)
        std = std_spread * (torch.max(x) - torch.min(x)) \
            if self.mask_init["method"] == "smoothgrad" else 0.

        sample_grad_yl = []
        sample_grad_yh = [[] for _ in range(self.J)]
        # All we need is to loop over the n_samples we need to get. For smoothgrad we do it n_times, and add noise
        # which we set to 0 in case we do not have smoothgrad method

        for i in range(n_samples):
            # x_noisy = self.x.clone().detach().requires_grad_(True).to(self.device)
            x_noisy = x.clone().detach().requires_grad_(True).to(self.device)
            # In case of saliency or x_grad we set the std to zero, so there is no noise added. E(X) is 0
            noise = torch.randn_like(self.x, dtype=torch.float32, device=self.device) * std
            x_noisy = x_noisy + noise

            assert x_noisy.requires_grad

            # softmax predictions
            yl_saliency, yh_saliency = self.forward_dwt(x_noisy.to(self.device))
            yl_saliency.requires_grad_(True)
            for s in yh_saliency:
                s.requires_grad_(True)

            x_saliency = self.inverse_dwt((yl_saliency, yh_saliency))
            assert x_saliency.requires_grad

            if self.mask_init["saliency_activation"] == "softmax_layer":
                softmax_predictions, _ = self._get_prediction(x_saliency, self.target_idx)
                # Get the negative log value of the softmax prediction
                nll = - torch.log(softmax_predictions)

                # Backpropagate
                nll.backward()

                # Add noisy gradients, if needed
                if self.mask_init["noisy_gradients"]:
                    gradient_noise = torch.randn_like(yl_saliency.grad, dtype=torch.float32, device=self.device) * \
                                     self.mask_init["gradient_noise"]
                    yl_saliency.grad = yl_saliency.grad + gradient_noise  # additive noise, for noisy gradient estimation

                if self.mask_init["method"] == "grad_x_input":
                    attr_yl, _ = torch.max(torch.abs(yl_saliency.grad * yl_saliency), dim=1)
                else:
                    attr_yl, _ = torch.max(torch.abs(yl_saliency.grad), dim=1)

                sample_grad_yl.append(attr_yl)

                for idx, s in enumerate(yh_saliency):

                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(s.grad, dtype=torch.float32,
                                                          device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        s.grad = s.grad + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        attr_yh, _ = torch.max(torch.abs(s.grad * s), dim=1)
                    else:
                        attr_yh, _ = torch.max(torch.abs(s.grad), dim=1)

                    sample_grad_yh[idx].append(attr_yh)

            elif self.mask_init["saliency_activation"] == "output_layer":
                # Output layer activation
                output_activation = self.model.forward(x_saliency)[:, self.target_idx]
                yl_saliency_grad = torch.autograd.grad(output_activation, yl_saliency, retain_graph=True)[0]

                # Add noisy gradients, if needed
                if self.mask_init["noisy_gradients"]:
                    gradient_noise = torch.randn_like(yl_saliency_grad, dtype=torch.float32, device=self.device) * \
                                     self.mask_init["gradient_noise"]
                    yl_saliency_grad = yl_saliency_grad + gradient_noise  # additive noise, for noisy gradient estimation

                if self.mask_init["method"] == "grad_x_input":
                    attr_yl, _ = torch.max(torch.abs(yl_saliency_grad * yl), dim=1)
                else:
                    attr_yl, _ = torch.max(torch.abs(yl_saliency_grad), dim=1)

                sample_grad_yl.append(attr_yl)

                for idx, s in enumerate(yh_saliency):
                    attr_yh = torch.autograd.grad(output_activation, s, retain_graph=True)[0]

                    # Add noisy gradients, if needed
                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(attr_yh, dtype=torch.float32,
                                                          device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        attr_yh = attr_yh + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        attr_yh, _ = torch.max(torch.abs(attr_yh * s), dim=1)
                    else:
                        attr_yh, _ = torch.max(torch.abs(attr_yh), dim=1)

                    sample_grad_yh[idx].append(attr_yh)

            else:
                raise ValueError(
                    "Please indicate a proper saliency activation option: choices: (softmax_layer, output_layer)."
                )

        # Take the mean over the samples we have and standardize it
        yl_mask_mean = torch.mean(torch.cat(sample_grad_yl), dim=0)
        yh_mask_mean = [
            torch.mean(torch.cat(yh_sample), dim=0) for yh_sample in sample_grad_yh
        ]

        yl_mask = self._standardize_saliency_map(yl_mask_mean.detach()).requires_grad_(True)
        yh_mask = [
            self._standardize_saliency_map(yh_coeff.detach()).requires_grad_(True) for yh_coeff in yh_mask_mean
        ]

    def _get_prediction(self, x, predicted_class_idx=None):
        logits_predictions = self.model.forward(x.to(self.device))
        softmax_predictions = F.softmax(logits_predictions, dim=1)

        if predicted_class_idx is None:
            predicted_class_idx = torch.argmax(softmax_predictions).item()
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]
        else:
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]

        return softmax_prediction_top_idx, predicted_class_idx

    def get_final_mask(self):
        final_mask_yh = []
        for mask in self.mask_yh:
            final_mask_yh.append(mask.detach())
        return [self.mask_yl.detach(), final_mask_yh]

    def _get_init_mask(self, yl, yh):
        yh_mask = []

        if self.mask_init["method"] == 'zeros':
            yl_mask = torch.zeros((yl.size(0), 1, *yl.shape[2:]),
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=True)

            for s in yh:
                yh_mask.append(torch.zeros((s.size(0), 1, *s.shape[2:]),
                                           dtype=torch.float32,
                                           device=self.device, requires_grad=True))

        elif self.mask_init["method"] == 'ones':
            yl_mask = torch.ones((yl.size(0), 1, *yl.shape[2:]),
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=True)

            for s in yh:
                yh_mask.append(torch.ones((s.size(0), 1, *s.shape[2:]),
                                          dtype=torch.float32,
                                          device=self.device, requires_grad=True))

        elif self.mask_init["method"] == "constant":
            yl_mask = torch.ones((yl.size(0), 1, *yl.shape[2:]),
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=True)

            with torch.no_grad():
                yl_mask *= self.mask_init["constant_value"]

            for s in yh:
                mask = torch.ones((s.size(0), 1, *s.shape[2:]),
                                          dtype=torch.float32,
                                          device=self.device, requires_grad=True)

                with torch.no_grad():
                    mask *= self.mask_init["method"]["constant_value"]

                yh_mask.append(mask)

        elif self.mask_init["method"] == 'uniform':
            # Creates a random mask initiated between 0 and 1
            yl_mask = torch.rand((yl.size(0), 1, *yl.shape[2:]),
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=True)

            for s in yh:
                yh_mask.append(torch.rand((s.size(0), 1, *s.shape[2:]),
                                          dtype=torch.float32,
                                          device=self.device, requires_grad=True))

        elif self.mask_init["method"] in ["saliency", "grad_x_input", "smoothgrad"]:

            std_spread = 0.15
            n_samples = 10 if self.mask_init["method"] == "smoothgrad" else 1
            x = self.x.clone().detach().requires_grad_(True).to(self.device)
            std = std_spread * (torch.max(x) - torch.min(x)) \
                if self.mask_init["method"] == "smoothgrad" else 0.

            sample_grad_yl = []
            sample_grad_yh = [[] for _ in range(self.J)]
            # All we need is to loop over the n_samples we need to get. For smoothgrad we do it n_times, and add noise
            # which we set to 0 in case we do not have smoothgrad method

            for i in range(n_samples):
                # x_noisy = self.x.clone().detach().requires_grad_(True).to(self.device)
                x_noisy = x.clone().detach().requires_grad_(True).to(self.device)
                # In case of saliency or x_grad we set the std to zero, so there is no noise added. E(X) is 0
                noise = torch.randn_like(self.x, dtype=torch.float32, device=self.device) * std
                x_noisy = x_noisy + noise

                assert x_noisy.requires_grad

                # softmax predictions
                yl_saliency, yh_saliency = self.forward_dwt(x_noisy.to(self.device))
                yl_saliency.requires_grad_(True)
                for s in yh_saliency:
                    s.requires_grad_(True)

                x_saliency = self.inverse_dwt((yl_saliency, yh_saliency))
                assert x_saliency.requires_grad

                if self.mask_init["saliency_activation"] == "softmax_layer":
                    softmax_predictions, _ = self._get_prediction(x_saliency, self.target_idx)
                    # Get the negative log value of the softmax prediction
                    nll = - torch.log(softmax_predictions)

                    # Backpropagate
                    nll.backward()

                    # Add noisy gradients, if needed
                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(yl_saliency.grad, dtype=torch.float32, device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        yl_saliency.grad = yl_saliency.grad + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        attr_yl, _ = torch.max(torch.abs(yl_saliency.grad * yl_saliency), dim=1)
                    else:
                        attr_yl, _ = torch.max(torch.abs(yl_saliency.grad), dim=1)

                    sample_grad_yl.append(attr_yl)

                    for idx, s in enumerate(yh_saliency):

                        if self.mask_init["noisy_gradients"]:
                            gradient_noise = torch.randn_like(s.grad, dtype=torch.float32,
                                                              device=self.device) * \
                                             self.mask_init["gradient_noise"]
                            s.grad = s.grad + gradient_noise  # additive noise, for noisy gradient estimation

                        if self.mask_init["method"] == "grad_x_input":
                            attr_yh, _ = torch.max(torch.abs(s.grad * s), dim=1)
                        else:
                            attr_yh, _ = torch.max(torch.abs(s.grad), dim=1)

                        sample_grad_yh[idx].append(attr_yh)

                elif self.mask_init["saliency_activation"] == "output_layer":
                    # Output layer activation
                    output_activation = self.model.forward(x_saliency)[:, self.target_idx]
                    yl_saliency_grad = torch.autograd.grad(output_activation, yl_saliency, retain_graph=True)[0]

                    # Add noisy gradients, if needed
                    if self.mask_init["noisy_gradients"]:
                        gradient_noise = torch.randn_like(yl_saliency_grad, dtype=torch.float32, device=self.device) * \
                                         self.mask_init["gradient_noise"]
                        yl_saliency_grad = yl_saliency_grad + gradient_noise  # additive noise, for noisy gradient estimation

                    if self.mask_init["method"] == "grad_x_input":
                        attr_yl, _ = torch.max(torch.abs(yl_saliency_grad * yl), dim=1)
                    else:
                        attr_yl, _ = torch.max(torch.abs(yl_saliency_grad), dim=1)

                    sample_grad_yl.append(attr_yl)

                    for idx, s in enumerate(yh_saliency):
                        attr_yh = torch.autograd.grad(output_activation, s, retain_graph=True)[0]

                        # Add noisy gradients, if needed
                        if self.mask_init["noisy_gradients"]:
                            gradient_noise = torch.randn_like(attr_yh, dtype=torch.float32,
                                                              device=self.device) * \
                                             self.mask_init["gradient_noise"]
                            attr_yh = attr_yh + gradient_noise  # additive noise, for noisy gradient estimation

                        if self.mask_init["method"] == "grad_x_input":
                            attr_yh, _ = torch.max(torch.abs(attr_yh * s), dim=1)
                        else:
                            attr_yh, _ = torch.max(torch.abs(attr_yh), dim=1)

                        sample_grad_yh[idx].append(attr_yh)

                else:
                    raise ValueError(
                        "Please indicate a proper saliency activation option: choices: (softmax_layer, output_layer)."
                    )

            # Take the mean over the samples we have and standardize it
            yl_mask_mean = torch.mean(torch.cat(sample_grad_yl), dim=0)
            yh_mask_mean = [
                torch.mean(torch.cat(yh_sample), dim=0) for yh_sample in sample_grad_yh
            ]

            yl_mask = self._standardize_saliency_map(yl_mask_mean.detach()).requires_grad_(True)
            yh_mask = [
                self._standardize_saliency_map(yh_coeff.detach()).requires_grad_(True) for yh_coeff in yh_mask_mean
            ]

        else:
            raise ValueError(
                "Need to either pass 'zeros', 'ones', 'random' or 'saliency' or 'grad_x_input' or "
                "'smoothgrad' for the initialization"
            )

        return yl_mask, yh_mask

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
            # ToDo: I can write it neater and I need to double-check this for shearletX
            if len(saliency_map.shape) == 2:
                saliency_map = saliency_map.unsqueeze(0)

            samples = saliency_map.shape[1] * saliency_map.shape[2]
            print(f"The number of samples used are {samples}")

            # Here we subsample, as otherwise it is very expensive!
            quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=min(10_000, samples),
                                                       subsample=min(10_000, samples))
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

    def _get_perturbation(self, yl, yh):
        """
        We only implement two methods, gaussian and uniform noise
        :return:tuple containing the perturbations
        """

        # First for the approximation coefficients
        mean_yl = torch.mean(yl)
        std_yl = torch.std(yl, correction=1)

        # Now for the detail coefficients
        mean_yh = [torch.mean(s) for s in yh]
        std_yh = [torch.std(s, correction=1) for s in yh]

        if self.perturbation_strategy == "gaussian":
            # Now we do the parameterization, using the reparameterization trick
            # Approximation level
            perturbation_yl = std_yl * torch.randn(self.batch_size, *yl.shape[1:],
                                                   dtype=torch.float32,
                                                   device=self.device,
                                                   requires_grad=False) + mean_yl

            # Detail coefficient
            perturbation_yh = [
                std_yh[i] * torch.randn(self.batch_size, *yh.shape[1:],
                                        dtype=torch.float32,
                                        device=self.device,
                                        requires_grad=False) + mean_yh[i]
                for i, yh in enumerate(yh)]

        elif self.perturbation_strategy == "uniform":
            perturbation_yl = torch.rand(self.batch_size, *yl.shape[1:],
                                         dtype=torch.float32,
                                         device=self.device,
                                         requires_grad=False) * (2 * std_yl) + (mean_yl - std_yl)

            # Detail coefficient
            perturbation_yh = [
                torch.rand(self.batch_size, *yh.shape[1:],
                           dtype=torch.float32,
                           device=self.device,
                           requires_grad=False) * (2 * std_yh[i]) + (mean_yh[i] - std_yh[i])
                for i, yh in enumerate(yh)]

        else:
            raise ValueError("Invalid perturbation choice")

        return perturbation_yl, perturbation_yh

    def _update_lambda(self, N):
        assert N != 0.
        log_delta_division_factor = np.log(self.delta) / N
        self.lambda_l1 = self.lambda_l1_final * np.exp(log_delta_division_factor)
        self.lambda_l2 = self.lambda_l2_final * np.exp(log_delta_division_factor)
        self.lambda_tv = self.lambda_tv_final * np.exp(log_delta_division_factor)