import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import QuantileTransformer


class ExtremaMaskExplainer:

    # Class-level constants for Smoothgrad mask initialization
    # We take the std_deviation_spread to be 0.15 (in fraction of the overall spread max-min)
    STD_SPREAD = 0.15
    N_SAMPLES = 25

    def __init__(self, prediction_model: nn.Module, perturbation_model: nn.Module,
                 lambda_1: float = 1., lambda_2: float = 1., lambda_compression: float = 0.,
                 lr: float = 0.01, device: str = "cpu") -> None:

        self.device = device
        # # For cuda perturbation model needs to be in train mode
        self._setup_prediction_model(prediction_model, device)

        # Alternative solution to the one here:
        # https://github.com/josephenguehard/time_interpret/blob/main/experiments/hmm/main.py
        # : Yet, this is much slower than the workaround used here.
        # Disable cudnn if using cuda accelerator.
        # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
        # for more information.
        # self.prediction_model = prediction_model.eval().to(self.device)
        # if device.type == 'cuda':
        #     torch.backends.cudnn.enabled = False

        self.perturbation_model = perturbation_model.to(self.device)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_compression = lambda_compression #Default term is set to zero
        self.lr = lr

    def attribute(self, x, iterations: int = 100, verbose: bool = True, mode: str = "preservation_game",
                  sparsity_method: str = "l1_norm", mask_initialization_method: str = "uniform",
                  transformation: str = "quantile_transformation", noisy_gradients: bool = False, gradient_noise: float = 0.01,
                  scaling: bool = True, baseline: str = "zero",
                  updated_perturbation_loss: bool = True, normalize_loss=False, use_softmax_target=False) -> tuple:

        """
        method to learn a mask to identify the most relevant parts of the input for the prediction
        :param x: input (s) to explain, dimension batch_size, sequence_length, feature_dim
        :param iterations: number of iterations to run algorithm with
        :param verbose: print loss to the stdout console
        :param mask_initialization_method: which method to use for mask initialization.
        Options: uniform, gradient_based, gradient_x_based
        :param transformation: which method to use to transform the mask initialization values at initialization.
        Only used when mask_initialization_method is not 'uniform'. Options: 'identity', 'sqrt'
        :param noisy_gradients: boolean, if true, we will add some noise to the gradients
        :param gradient_noise: std of the noise to be added.
        :param scaling: Boolean. If true, we scale the gradient signal to be in the [0, 1] range.
        :param sparsity_method: which method to induce sparse mask. either l1_norm or log_energy (gaussian entropy)
        :param mode: either preservation_game or deletion_game
        :param baseline: reference_value for the deletion game.
        Choices: zero (as implemented in the original paper) or mean
        :param updated_perturbation_loss: use a different perturbation loss as implemented proposed here
        TMLR paper (2024): https://openreview.net/pdf?id=fCNqD2IuoD

        preservation_game: preserve as much of the original prediction, while masking as much as possible.
        deletion_game: delete as little as possible, with the biggest impact

        :return: tuple (mask, mask_history). the learned masks of shape, batch_size, sequence_length, feature_dim and
        a list of masks at each iteration steps
        """

        assert len(x.shape) == 3, "Input tensor must have 3 dimensions [batch_size, sequence_len, feature_dim]"

        x = x.to(self.device).requires_grad_(False)

        mask = self._initialize_mask(x, mask_initialization_method,
                                     transformation, mode, noisy_gradients, gradient_noise, scaling)

        if normalize_loss:
            number_coefficients = mask.view(mask.size(0), -1).size(-1)

        self._configure_optimizer(mask)

        # We will need a mask history to retrieve different performance metrics at certain steps
        mask_history = []

        # # We start at iteration 0 as the plots will also start at iteration 0, so we are in line with that
        attr = mask.detach().clone() if mode == "preservation_game" else 1 - mask.detach().clone()
        mask_history.append(attr)

        with torch.no_grad():
            if mode == "preservation_game":
                prediction_target = self.prediction_model(x)
                softmax_prediction_target = F.softmax(prediction_target, dim=1)

            else:
                reference_value = self._get_baseline(x, baseline)
                prediction_target = self.prediction_model(reference_value)

        for _ in tqdm(range(1, iterations+1), desc=f"Please wait we train the masks "
                                                   f"with init {mask_initialization_method}"):

            # Step 1: calculate the perturbation / background
            perturbation = self.perturbation_model(x)
            # Step 2: calculate the compressed representation. These are dimension: B x T
            x_distorted = mask * x + (1 - mask) * perturbation

            # Step 3: calculate the prediction for distorted representation; dimension: B x T
            distorted_predictions = self.prediction_model(x_distorted)
            distorted_predictions_softmax = F.softmax(distorted_predictions, dim=1)

            # Step 4: calculate the distortion
            # MSE loss function outperforms cross entropy loss function:
            # See TMLR paper (2024): https://openreview.net/pdf?id=fCNqD2IuoD
            # distortion = ((distorted_predictions - prediction_target) ** 2).sum(axis=1)
            if len(distorted_predictions.shape) == 3:
                # Sum across the states and dimensions!
                if use_softmax_target:
                    distortion = ((distorted_predictions_softmax - softmax_prediction_target) ** 2).sum(axis=(1,2))
                else:
                    distortion = ((distorted_predictions - prediction_target) ** 2).sum(axis=(1, 2))
            else:
                if use_softmax_target:
                    distortion = ((distorted_predictions_softmax - softmax_prediction_target) ** 2).sum(axis=1)
                else:
                    distortion = ((distorted_predictions - prediction_target) ** 2).sum(axis=1)

            # Sparsity_loss. mask is of dimension B x T x D, resulting
            mask_value = mask if mode == "preservation_game" else (1 - mask)

            if sparsity_method == "l1_norm":
                l1_loss = mask_value.abs().sum(axis=(1, 2))

            elif sparsity_method == "log_energy":
                log_energy = torch.log(torch.abs(mask_value + 1e-15) ** 2)

                # Calculate l1_loss
                l1_loss = log_energy.sum(axis=(1, 2))

            else:
                raise ValueError(f"Unknown sparsity method {sparsity_method}. Options are: 'l1_norm', 'log_energy'")


            # Perturbation_loss, we take the l1 norm across Batches as described in the paper
            if mode == "preservation_game":
                perturbation_loss = perturbation.abs().sum(axis=(1, 2))

            else:
                if updated_perturbation_loss:
                    # We implement the improved deletion game formulation
                    # See this reproducibility paper: https://openreview.net/pdf?id=nPZgtpfgIx
                    perturbation_loss = ((perturbation - x) ** 2).sum(axis=(1, 2))
                else:
                    perturbation_loss = perturbation.abs().sum(axis=(1, 2))

            if normalize_loss:
                l1_loss /= number_coefficients
                perturbation_loss /= number_coefficients

            # Compression term m * x
            compression = (mask * x).abs().sum(axis=(1, 2))

            total_loss = distortion + self.lambda_1 * l1_loss + self.lambda_2 * perturbation_loss + \
                self.lambda_compression * compression

            # # Get the average over each sample to explain
            mean_loss = total_loss.mean()

            # Backward and optimize
            self.optimizer.zero_grad()
            mean_loss.backward(retain_graph=True)
            self.optimizer.step()

            if verbose:
                tqdm.write(f"\nIteration loss: {mean_loss.item()}")

            with torch.no_grad():
                mask.clamp_(0, 1)

            # Store the masks as we need them later for plotting each across different iteration steps
            # We start at iteration 0 as the plots will also start at iteration 0, so we are in line with that
            attr = mask.detach().clone() if mode == "preservation_game" else 1 - mask.detach().clone()
            mask_history.append(attr)

        # In deletion game, we want to delete the parts of the image with the biggest impact,
        # Hence, those are set to zero, so true meaning is 1-mask
        attr = mask if mode == "preservation_game" else 1 - mask

        return attr, mask_history

    def _setup_prediction_model(self, prediction_model, device) -> None:
        """
        Adjusts the model's Dropout and BatchNorm1d layers if the prediction model uses it, as Cuda has issues with it
        - Sets Dropout layers' probability to 0.
        - Sets BatchNorm1d layers to evaluation mode.

        Parameters:
        - model (nn.Module): The prediction network model to adjust.
        """

        self.prediction_model = prediction_model.train().to(device)

        for module in self.prediction_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
            elif isinstance(module, nn.BatchNorm1d):
                module.eval()
                module.track_running_stats = False

        # Checking that it works correctly
        for name, module in self.prediction_model.named_modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0, f"Dropout layer {name} not set to 0, found {module.p}"
            elif isinstance(module, nn.BatchNorm1d):
                assert not module.training, f"BatchNorm1d layer {name} not in eval mode"
                assert not module.track_running_stats, f"BatchNorm1d layer {name} is tracking running stats"

    def _initialize_mask(self, x: torch.Tensor,
                         mask_initialization_method: str,
                         transformation: str, mode: str,
                         noisy_gradients: bool,
                         gradient_noise: float,
                         scaling: bool,
                         ) -> torch.Tensor:
        """
        Initializes the mask according to the mask initialization method provided in the constructor
        :param x: input tensor which determines the shape of the mask
        :param mask_initialization_method: which method to use for mask initialization.
        Options: uniform, gradient_based, gradient_x_based
        :param transformation: which method to use to transform the mask initialization values at initialization.
        Only used when mask_initialization_method is not 'uniform'. Options: 'identity', 'sqrt'
        :param mode: either preservation game or deletion_game
        :return: nn.Parameter with the mask coefficients initialized
        """

        if mask_initialization_method in ("ones", "uniform"):
            # 0.5 is the expected value of a uniform (0,1) distribution
            mu = 0.5 if mask_initialization_method == "uniform" else 1.
            mask = nn.Parameter(mu * torch.ones(*x.shape, device=self.device), requires_grad=True)

        elif mask_initialization_method == "random_uniform":
            values = torch.randn(*x.shape, device=self.device)
            mask = nn.Parameter(values, requires_grad=True)

        elif mask_initialization_method == "zeros":
            mask = nn.Parameter(torch.zeros(*x.shape, device=self.device), requires_grad=True)

        elif mask_initialization_method in ("gradient_based", "gradient_x_based", "smoothgrad"):
            # See here:
            # https://github.com/pytorch/pytorch/issues/10006

            input_saliency = x.clone().requires_grad_(True).to(self.device)
            if mask_initialization_method == "smoothgrad":
                std = self.STD_SPREAD * \
                      (torch.amax(x, dim=(1, 2), keepdim=True) - torch.amin(x, dim=(1, 2), keepdim=True))

                attr_samples = []

                for _ in range(self.N_SAMPLES):
                    noise = torch.randn(*x.shape, dtype=torch.float32, device=self.device) * std
                    input_saliency_noisy = input_saliency + noise
                    output_activation = self.prediction_model(input_saliency_noisy)
                    grad_outputs = torch.ones_like(output_activation, device=self.device)
                    attr = torch.autograd.grad(output_activation, input_saliency_noisy, grad_outputs=grad_outputs)[0]

                    if noisy_gradients:
                        noise = torch.randn_like(attr, dtype=torch.float32, device=self.device) * gradient_noise
                        attr = attr + noise

                    attr_samples.append(attr)

                # We stack the data at dim = 1, so we then have B x smoothgrad_samples x Sequence_len x D
                # Then we can take the mean across smoothgrad_samples and get B x Sequence_len x D
                attr = torch.mean(torch.stack(attr_samples, dim=1), dim=1)

            else:
                output_activation = self.prediction_model(input_saliency)
                if output_activation.shape[1] == 2:
                    # In we have two states, we take the torch.max
                    output_activation = torch.max(output_activation, dim=1)[0].reshape(-1, 1)

                grad_outputs = torch.ones_like(output_activation, device=self.device)
                attr = torch.autograd.grad(output_activation, input_saliency, grad_outputs=grad_outputs)[0]

                if noisy_gradients:
                    noise = torch.randn_like(attr, dtype=torch.float32, device=self.device) * gradient_noise
                    attr = attr + noise

                if mask_initialization_method == "gradient_x_based":
                    attr = attr * input_saliency

            # Important note
            # Important: our gradients follow a laplace distribution:
            # Has been observed also in graphs see TAGE paper:
            # https://arxiv.org/abs/2202.08335
            # We can use the quantile transformation:
            # https://machinelearningmastery.com/quantile-transforms-for-machine-learning/

            attr = self._get_transformed_gradient_values(attr, transformation, scaling)

            if mode == "deletion_game":
                # Rationale:
                # In deletion game, I need to set the important features to zero that are important to get the
                # prediction close to baseline (uninformative one).
                attr = 1 - attr

            mask = nn.Parameter(attr, requires_grad=True)

        else:
            raise ValueError("Please provide a valid initialization scheme. "
                             "Options: 'uniform', 'ones', 'gradient-based', 'gradient_x_based', 'smoothgrad'")

        return mask

    def _configure_optimizer(self, mask: torch.Tensor) -> None:
        """
        Configures the optimizer
        :param mask: the mask which will be optimizer
        :return: None
        """
        parameters = [{"params": mask}]
        parameters += [{"params": self.perturbation_model.parameters()}]
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr)

    def _get_baseline(self, x: torch.Tensor, baseline: str) -> torch.tensor:
        """
        Gets the baseline tensor for deletion game
        :param x: input tensor
        :param baseline: baseline method, either zero or mean
        :return: baseline tensor
        """
        if baseline == "zero":
            reference_value = torch.zeros_like(x).to(self.device)
        elif baseline == "mean":
            reference_value = (torch.ones(*x.shape) * x.mean()).to(self.device)
        else:
            raise ValueError("Invalid baseline method. Options either 'zero' or 'mean'")

        return reference_value

    @staticmethod
    def _min_max_scale_attr(attr: torch.Tensor) -> torch.Tensor:
        """
        Static method to min-max scale the attribute.
        This is also used in the reference paper:
        Check: https://github.com/josephenguehard/time_interpret/blob/main/tint/metrics/white_box/base.py

        :param attr: mask coefficients that should be min-max scaled
        :return: min-max scaled mask coefficients
        """

        assert len(attr.shape) == 3, "We expect the attr values to be of dim batch, seq, feature_dim! "
        min_per_mask = torch.amin(attr, dim=(1, 2), keepdim=True)
        max_per_mask = torch.amax(attr, dim=(1, 2), keepdim=True)

        attr_min_max_scaled = (attr - min_per_mask) / (max_per_mask - min_per_mask + 1e-5)

        # Assertions to check the scaling
        assert torch.all(attr_min_max_scaled >= 0.), "Min-max scaled values should be >= 0."
        assert torch.all(attr_min_max_scaled <= 1.), "Min-max scaled values should be <= 1."

        return attr_min_max_scaled

    def _get_transformed_gradient_values(self, attr, transformation, scaling):
        if transformation == "sqrt":
            attr = torch.sqrt(attr.abs())

        elif transformation == "identity":
            attr = attr.abs()

        elif transformation == "quantile_transformation":
            samples = attr.shape[1] * attr.shape[2]
            quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=samples,
                                                       subsample=None)
            transformed_attr_list = [quantile_transformer.fit_transform(mask.reshape(-1, 1).cpu()) for mask in attr]
            attr = torch.tensor(np.array(transformed_attr_list), dtype=torch.float32,
                                device=self.device, requires_grad=True).reshape(*attr.shape)

        # Now I need to min-max scale it, so it is in range(0, 1)
        if transformation in ("sqrt", "identity") and scaling:
            attr = self._min_max_scale_attr(attr)

        return attr

