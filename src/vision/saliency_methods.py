#Saliency method clas
import torch
import numpy as np
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel, LayerGradCam, LayerAttribution, LRP
from utils.vision.helper import time_it
from sklearn.preprocessing import QuantileTransformer


class Saliency_Explainer:
    """
    Class that implements various saliency explanations using the CaptumAI method
    """
    def __init__(self, model, device='cpu', method='IG', qtf_standardization=False):

        self.model = model.to(device)
        self.device = device
        self.method = method
        self.qtf_standardization = qtf_standardization

    def _get_prediction(self, x, predicted_class_idx=None):
        logits_predictions = self.model.forward(x.to(self.device))
        softmax_predictions = F.softmax(logits_predictions, dim=1)

        if predicted_class_idx is None:
            predicted_class_idx = torch.argmax(softmax_predictions).item()
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]
        else:
            softmax_prediction_top_idx = softmax_predictions[:, predicted_class_idx]

        return softmax_prediction_top_idx, predicted_class_idx

    def _get_cp_1_score(self, input):
        original_softmax_prediction, predicted_class = self._get_prediction(input)
        distorted_softmax_predictions, _ = self._get_prediction(self.distorted_input.to(torch.float32), predicted_class)

        # First retained probability
        retained_class_probability = torch.mean(distorted_softmax_predictions / distorted_softmax_predictions, dim=0)

        # Get the masked information
        pixel_masked_information = (self.attr.detach() * input).abs().sum().item()
        # pixel_original_information = x.view(1, -1).detach().abs().sum().item()
        pixel_original_information = input.reshape(1, -1).detach().abs().sum().item()

        retained_information_pixel = pixel_masked_information / pixel_original_information
        cp_pixel = retained_class_probability / retained_information_pixel

        print(f"cp pixel score is: {cp_pixel}")
        self.cp_pixel_score = cp_pixel

    def __call__(self, input, target_idx=None, standardize=True):

        self.input_saliency = input.clone().detach().requires_grad_(True).float().to(self.device)
        # We can calculate the baseline already here
        mean = torch.mean(self.input_saliency).unsqueeze(0).to(self.device)
        std = torch.std(self.input_saliency, correction=1).unsqueeze(0).to(self.device)
        self.baseline = std * torch.rand(*self.input_saliency.shape, dtype=torch.float32, device=self.device) + mean

        if target_idx is None:
            with torch.no_grad():
                target_idx = torch.argmax(self.model.forward(self.input_saliency), dim=-1)

        if self.method == "IG" or self.method == "IntegratedGradients":
            return self._get_integrated_gradient_saliency(target_idx=target_idx, standardize=standardize)

        elif self.method == "SG" or self.method == "SmoothGrad":
            return self._get_smoothgrad_saliency(target_idx=target_idx, standardize=standardize)

        elif self.method == "VarGrad":
            return self._get_vargrad_saliency(target_idx=target_idx, standardize=standardize)

        elif self.method == "SquareGrad":
            return self._get_squaregrad_saliency(target_idx=target_idx, standardize=standardize)

        elif self.method == "Saliency":
            return self._get_saliency(target_idx=target_idx, standardize=standardize)

        elif self.method == "GradInput":
            return self._get_grad_input(target_idx=target_idx, grad_input=True, standardize=standardize)

        elif self.method == "GradCAM":
            return self._get_gradcam(target_idx=target_idx, standardize=standardize)

        elif self.method == "LRP":
            return self._get_lrp(target_idx=target_idx, standardize=standardize)

        else:
            raise ValueError("Please indicate a proper method! Choice: 'IG', 'SG', 'Saliency', 'GradCAM', 'LRP'")

    @time_it(method_name="IG")
    def _get_integrated_gradient_saliency(self, target_idx, standardize=True):

        # If provided with a target idx, change the argmax default target to the one provided
        # To be in line with the paper, we use the uniform perturbation as the baseline
        mean = torch.mean(self.input_saliency).unsqueeze(0).to(self.device)
        std = torch.std(self.input_saliency, correction=1).unsqueeze(0).to(self.device)
        baseline = std * torch.rand(*self.input_saliency.shape, dtype=torch.float32, device=self.device) + mean

        grad_method = IntegratedGradients(self.model)
        attr = grad_method.attribute(self.input_saliency, baselines=baseline, target=target_idx, n_steps=50)[0]

        torch.cuda.empty_cache()

        #Maximum across each channel
        attr, _ = torch.max(attr, dim=0)

        if standardize:
            self.attr = self._standardize_saliency_map(attr)

        self.distorted_input = self.attr * self.input_saliency + (1 - self.attr) * baseline

        # get CP-pixel score
        self._get_cp_1_score(self.input_saliency)

        # #returns it on cpu in case tensor is on cpu
        return self.attr.cpu()
        # return self.final_explanation.squeeze().cpu() if self.qtf_standardization else attr.cpu()

    @time_it(method_name="SG")
    def _get_smoothgrad_saliency(self, target_idx, standardize=True):
        grad_ = Saliency(self.model.forward)
        grad_method = NoiseTunnel(grad_)
        # attr = grad_method.attribute(self.input_saliency, target=target_idx, stdevs=0.1)[0]
        attr = grad_method.attribute(self.input_saliency, target=target_idx)[0]

        torch.cuda.empty_cache()

        # Maximum across each channel
        attr, _ = torch.max(attr, dim=0)

        if standardize:
            attr = self._standardize_saliency_map(attr)

        #Return it to cpu in case it is on gpu
        return attr.cpu()

    @time_it(method_name="VarGrad")
    def _get_vargrad_saliency(self, target_idx, standardize=True):
        grad_ = Saliency(self.model.forward)
        grad_method = NoiseTunnel(grad_)
        # attr = grad_method.attribute(self.input_saliency, target=target_idx, stdevs=0.1, nt_type="vargrad")[0]
        attr = grad_method.attribute(self.input_saliency, target=target_idx, nt_type="vargrad")[0]

        torch.cuda.empty_cache()

        # Maximum across each channel
        attr, _ = torch.max(attr, dim=0)

        if standardize:
            attr = self._standardize_saliency_map(attr)

        #Return it to cpu in case it is on gpu
        return attr.cpu()

    @time_it(method_name="SquareGrad")
    def _get_squaregrad_saliency(self, target_idx, standardize=True):
        grad_ = Saliency(self.model.forward)
        grad_method = NoiseTunnel(grad_)
        # attr = grad_method.attribute(self.input_saliency, target=target_idx, stdevs=0.1,
        #                              nt_type="smoothgrad_sq")[0]
        attr = grad_method.attribute(self.input_saliency, target=target_idx,
                                     nt_type="smoothgrad_sq")[0]

        torch.cuda.empty_cache()

        # Maximum across each channel
        attr, _ = torch.max(attr, dim=0)

        if standardize:
            attr = self._standardize_saliency_map(attr)

        #Return it to cpu in case it is on gpu
        return attr.cpu()

    @time_it(method_name="Saliency")
    def _get_saliency(self, target_idx, standardize=True):
        grad = Saliency(self.model.forward)
        attr = grad.attribute(self.input_saliency, target=target_idx)[0]

        attr, _ = torch.max(attr.abs(), dim=0)

        if standardize:
            attr = self._standardize_saliency_map(attr)

        return attr.cpu()

    @time_it(method_name="GradInput")
    def _get_grad_input(self, target_idx, grad_input=False, standardize=True):
        grad = Saliency(self.model.forward)
        attr = grad.attribute(self.input_saliency, target=target_idx)[0]

        if grad_input:
            attr = attr * self.input_saliency

        attr, _ = torch.max(attr.abs(), dim=0)

        if standardize:
            attr = self._standardize_saliency_map(attr)

        return attr.cpu()

    @time_it(method_name="GradCAM")
    def _get_gradcam(self, target_idx, standardize=True):
        # Get the last convolutional layer
        last_conv_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model")

        grad_cam = LayerGradCam(self.model, last_conv_layer)
        attr = grad_cam.attribute(self.input_saliency, target=target_idx)[0]
        
        # Upsample attributions to match input size
        attr = LayerAttribution.interpolate(attr.unsqueeze(0), self.input_saliency.shape[2:])[0]

        # Take the maximum across channels
        attr, _ = torch.max(attr, dim=0)

        if standardize:
            self.attr = self._standardize_saliency_map(attr)

        self.distorted_input = self.attr * self.input_saliency + (1 - self.attr) * self.baseline

        # get CP-pixel score
        self._get_cp_1_score(self.input_saliency)

        return self.attr.cpu()

    @time_it(method_name="LRP")
    def _get_lrp(self, target_idx, standardize=True):
        """
        Computes Layer-wise Relevance Propagation attribution for the input.
        """
        # First standardize the input using the first layer (Standardizer)
        standardized_input = self.model[0](self.input_saliency)
        
        # Create temporary model using only the base model (second part of Sequential)
        temp_model = self.model[1]
        
        lrp = LRP(temp_model)
        attr = lrp.attribute(standardized_input, target=target_idx)[0]
        
        torch.cuda.empty_cache()
        
        # Maximum across each channel
        attr, _ = torch.max(attr, dim=0)
        
        if standardize:
            attr = self._standardize_saliency_map(attr)
        
        return attr.cpu()

    def _standardize_saliency_map(self, saliency_map):

        if self.qtf_standardization:
            if len(saliency_map.shape) == 2:
                saliency_map = saliency_map.unsqueeze(0)

            samples = saliency_map.shape[1] * saliency_map.shape[2]
            print(f"The number of samples used are {samples}")

            # Here we subsample, as otherwise it is very expensive!
            quantile_transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=min(10_000, samples),
                                                       subsample=min(10_000, samples))
            transformed_attr_list = [
                quantile_transformer.fit_transform(mask.reshape(-1, 1).detach().cpu()) for mask in saliency_map
            ]
            saliency_standardized = torch.tensor(np.array(transformed_attr_list), dtype=torch.float32,
                                device=self.device).reshape(*saliency_map.shape).squeeze()

        else:
            saliency_map_min = torch.min(saliency_map)
            saliency_map_max = torch.max(saliency_map)

            saliency_standardized = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        return saliency_standardized.unsqueeze(0)

# Code adapted from:
# https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py
class NoiseGrad(Saliency_Explainer):

    def __init__(
            self,
            model,
            device,
            method,
            n: int = 25,
            mean: float = 1.,
            std: float = 0.2,
            noise_type: str = "multiplicative",
    ):

        if method not in {"IG", "SG", "Saliency"}:
            raise ValueError("Please indicate a proper method! Choice: 'IG', 'SG', 'Saliency'")

        super().__init__(
            model,
            device,
            method,
        )

        self.n = n
        self.mean = mean
        self.std = std
        self.model_state = self.model.state_dict()

        if noise_type not in {"multiplicative", "additive"}:
            raise ValueError("Only 'multiplicative' and 'additive' as options")

        self.noise_type = noise_type
        self.distribution = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

    def _sample(self):
        with torch.no_grad():
            for layer in self.model.parameters():
                if self.noise_type == "multiplicative":
                    layer.mul_(
                        self.distribution.sample(layer.size()).to(layer.device)
                    )
                else:
                    layer.add_(
                        self.distribution.sample(layer.size()).to(layer.device)
                    )

    def _reset_model_weights(self):
        self.model.load_state_dict(self.model_state)

    @time_it(method_name="NoiseGrad")
    def __call__(self, input, target_idx=None, standardize=True):

        explanations = []
        for i in range(self.n):
            self._sample()
            self.input_saliency = input.clone().detach().requires_grad_(True).float().to(self.device)

            if target_idx is None:
                with torch.no_grad():
                    target_idx = torch.argmax(self.model.forward(self.input_saliency), dim=-1)

            if self.method == "IG":
                explanations.append(self._get_integrated_gradient_saliency(target_idx=target_idx,
                                                                           standardize=standardize))

            elif self.method == "SG":
                explanations.append(self._get_smoothgrad_saliency(target_idx=target_idx, standardize=standardize))

            elif self.method == "Saliency":
                explanations.append(self._get_saliency(target_idx=target_idx, standardize=standardize))

        # We need to make sure that we use the original weights for downstream tasks
        self._reset_model_weights()

        return torch.mean(torch.stack(explanations, dim=0), dim=0)


