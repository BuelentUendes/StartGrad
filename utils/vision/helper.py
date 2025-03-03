#Helper function module especially for evaluation

import math
import os
import time
from functools import wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from utils.general.helper_path import PERFORMANCE_TRADEOFF_PATH, PERFORMANCE_STARTGRAD_PATH

# Standard measures for mean and std for the transformation for IMAGENET model_architecture
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

SIZE = 256
VIT_SIZE = 224
VIT_SWIN_SIZE = 224

def time_it(method_name=None):
    """
    Decorator to measure the execution time of a function and allow for a custom method name.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Use custom method name if provided, otherwise default to func.__name__
            name = method_name if method_name else func.__name__

            with open(os.path.join(PERFORMANCE_TRADEOFF_PATH, f"Execution_time_{name}.txt"), "a") as f:
                f.write(f"\n{elapsed_time:.4f}")

            return result
        return wrapper
    return decorator


def time_it_QTF(method_name=None):
    """
    Decorator to measure the execution time of the QTF transformation and allow for a custom method name.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Use custom method name if provided, otherwise default to func.__name__
            name = method_name if method_name else func.__name__

            with open(os.path.join(PERFORMANCE_STARTGRAD_PATH, f"Execution_time_{name}.txt"), "a") as f:
                f.write(f"\n{elapsed_time:.4f}")

            return result
        return wrapper
    return decorator


def torchsheardec2D(torch_X,  torch_shearlets, device="cpu"):
    """Shearlet Decomposition function."""
    Xfreq = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(torch_X))).to(device)
    coeffs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(torch.einsum('bij,bijk->bijk',
                                                                                 Xfreq, torch_shearlets)), dim=(-3, 2)))
    return torch.real(coeffs)


def torchshearrec2D(torch_coeffs,  torch_shearlets, dualFrameWeights, dtype=torch.float32, device="cpu"):
    """Shearlet Reconstruction function."""
    torchdualFrameWeights = torch.Tensor(dualFrameWeights).type(dtype).to(device)
    torch_coeffs_freq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch_coeffs)))
    Xfreq = torch.sum(torch_coeffs_freq*torch_shearlets.permute(0, 3, 1, 2), dim=1).to(device)

    return torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Xfreq/torchdualFrameWeights))))


def preprocess_imagenet_data(input_image, model_name="resnet18"):
    """
    Preprocess images according to model requirements
    :param input_image:
    :param model_name:
    :return:
    """
    if model_name == "vit_base":
        # ViT specific preprocessing
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # We follow the pipeline exactly as stated in PyTorch
            # Yet, we do not resize to 224 x 224 but instead leave it as 256
            # The rest of the experiments do it as well
            transforms.Resize(size=(SIZE, SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(VIT_SIZE),  # Here it would normally crop 224
        ])

    elif model_name == "swin_base":
        # ViT Swin specific preprocessing
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # We follow the pipeline exactly as stated in PyTorch
            # Yet, we do not resize to 224 x 224 but instead leave it as 256
            # The rest of the experiments do it as well
            # pipeline: https://pytorch.org/vision/main/models/generated/torchvision.models.swin_b.html#torchvision.models.swin_b
            transforms.Resize(size=(238, 238), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(VIT_SIZE),  # Here it would normally crop 224
        ])

    elif model_name == "swin_t":
        # Here we use the efficient swin_v2_t architecture
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # We follow the pipeline exactly as stated in PyTorch
            # Yet, we do not resize to 224 x 224 but instead leave it as 256
            # The rest of the experiments do it as well
            # pipeline: https://pytorch.org/vision/main/models/generated/torchvision.models.swin_v2_t.html#torchvision.models.swin_v2_t
            transforms.Resize(size=(260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(SIZE),  # Here it would normally crop 256
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(SIZE, SIZE)),
        ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


class ContrastiveTransformations(object):
    """
    Code from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    """
    def __init__(self, base_transforms, n_views=16):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        # Basically we keep the original one in as the original one
        return [x if i == 0 else self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.45,
                               contrast=0.45,
                               saturation=0.45,
                               hue=0.3)
    ], p=0.8),
    transforms.RandomResizedCrop(size=SIZE, scale=(0.6, 0.9), ratio=(1., 1.)),
    transforms.GaussianBlur(kernel_size=9)
])


def get_probability_tensor_from_mask(mask):
    probability_class_1 = mask.flatten()
    probability_tensor = torch.zeros((probability_class_1.shape[0], 2))
    probability_tensor[:, 0] = 1 - probability_class_1
    probability_tensor[:, 1] = probability_class_1

    return probability_tensor


# Gumbel softmax functions
def get_gumbel_samples(sample_size, k, probability_tensor, mu=0., beta=1.):
    uniform_samples = torch.rand((sample_size, k))
    gumbel_samples = mu - beta * torch.log(-torch.log(uniform_samples))
    # Get the classes:
    sampled_values = torch.log(probability_tensor) + gumbel_samples
    return sampled_values


def gumbel_softmax(sample_size, k, probability_tensor, temperature=0.1, mu=0., beta=1., softmax=False):
    samples = get_gumbel_samples(sample_size, k, probability_tensor, mu=mu, beta=beta)
    if softmax:
        y = F.softmax(samples / temperature, dim=-1).float()
    else:
        y = torch.argmax(samples, dim=-1).float()
    return y


# Code adapted from: https://github.com/skmda37/ShearletX/blob/main/code/imagenet_utils/main.py
class Standardizer(nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean, device=device, requires_grad=False)
        self.std = torch.tensor(std, device=device, requires_grad=False)

    def forward(self, x):
        x = x - self.mean.reshape(self.mean.size(0), 1, 1)
        x = x / self.std.reshape(self.std.size(0), 1, 1)
        return x


# Code taken from: https://github.com/skmda37/ShearletX/blob/main/code/imagenet_utils/main.py
def get_model(model_name="resnet18", device="cpu", normalize_input=True):

    if model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        neural_net = models.vgg16(weights=weights).eval().to(device)

    elif model_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT
        neural_net = models.vgg19(weights=weights).eval().to(device)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        neural_net = models.resnet18(weights=weights).eval().to(device)

    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        neural_net = models.resnet34(weights=weights).eval().to(device)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        neural_net = models.resnet50(weights=weights).eval().to(device)

    elif model_name == "mobilenetv3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        neural_net = models.mobilenet_v3_small(weights=weights).eval().to(device)

    elif model_name == "vit_base":
        weights = models.ViT_B_16_Weights.DEFAULT
        neural_net = models.vit_b_16(weights=weights).eval().to(device)

    elif model_name == "swin_base":
        weights = models.Swin_B_Weights.DEFAULT
        neural_net = models.swin_b(weights=weights).eval().to(device)

    elif model_name == "swin_t":
        weights = models.Swin_V2_T_Weights.DEFAULT
        neural_net = models.swin_v2_t(weights=weights).eval().to(device)

    else:
        raise ValueError("Please indicate a model name that is implemented!")

    for parameter in neural_net.parameters():
        parameter.requires_grad = False

    if normalize_input:
        neural_net = nn.Sequential(Standardizer(NORM_MEAN, NORM_STD, device), neural_net)

    return neural_net


def get_distortion(model, original_input, distorted_input, argmax=True, return_prediction_prob=False, device="cpu"):
    """
    Calculates the squared error between the original softmax prob and the distorted img
    :param model: pre-trained model
    :param original_input: original image
    :param distorted_input: distorted image
    :param argmax: boolean indicating if we want L2 norm or only considering argmax (in the paper, they used a L2 norm)
    :param return prediction_prob: if true, then we return not the distortion, but the resulting prediciton prob
    :return: squared error in terms of max probability that occurs due to masking out the most-important features
    """
    if argmax:
        original_prob = torch.max(F.softmax(model.forward(original_input.to(device)), dim=1)).cpu().item()
        idx_max_prob = torch.argmax(model.forward(original_input.to(device)), -1)
        distorted_prob = F.softmax(model.forward(distorted_input.to(device)), dim=1)[:, idx_max_prob].cpu().item()
        distortion = ((original_prob - distorted_prob)**2)

    else:
        original_prob = F.softmax(model.forward(original_input.to(device)), dim=1).cpu()
        distorted_prob = F.softmax(model.forward(distorted_input.to(device)), dim=1).cpu()
        distortion = ((original_prob - distorted_prob)**2).sum().item()

    #Lastly, we need to take the square root of the squared sum for the L2 norm
    if return_prediction_prob:
        return distorted_prob
    else:
        return distortion


def randomize_relevant_components(saliency_explanation, img, percentage, background='uniform', randomize_ascending=False):
    """
    Randomizes non-relevant parts
    :param saliency_explanation: original explanation
    :param img: original image
    :param percentage: percentage to keep, if p=0.1, then 0.9 will be randomized, if p=0.2,  then 0.8 will be shuffled etc.
    :param background: background distribution that is needed for imputations. Choice (mean, gaussian, zero, ones, uniform). Default: mean
    :return: distorted image
    0 -> 100% randomized
    0.2 -> 80% randomized
    """
    if not torch.is_tensor(img):
        img = torch.tensor(img)

    if len(saliency_explanation.shape) == 2:
        saliency_explanation = saliency_explanation.unsqueeze(0)

    # Reshape the tensors
    saliency_explanation_reshaped = saliency_explanation.reshape(-1, saliency_explanation.shape[1] * saliency_explanation.shape[2]).\
        detach().cpu().numpy()

    # Sort values in a descending manner
    sorted_saliencies = np.sort(saliency_explanation_reshaped, axis=1)[:, ::-1]

    # Get the number of values that
    top_k = int(sorted_saliencies.shape[1] * percentage)
    threshold = sorted_saliencies[:, top_k - 1] if percentage != 0.0 else sorted_saliencies[:, 0]

    if randomize_ascending:
        mask = torch.tensor((saliency_explanation_reshaped <= threshold[:, None]).astype(float),
                            dtype=torch.float32).view_as(saliency_explanation)

    else:
        mask = torch.tensor((saliency_explanation_reshaped >= threshold[:, None]).astype(float),
                            dtype=torch.float32).view_as(saliency_explanation)

    if background == "mean":
        background_value = torch.mean(img)

    elif background == "zero":
        background_value = torch.zeros_like(img)

    elif background == "ones":
        background_value = torch.ones_like(img)

    elif background == "gaussian":
        mean = torch.mean(img)
        std = torch.std(img, correction=1)
        background_value = std * torch.randn_like(img) + mean

    elif background == "uniform":
        mean = torch.mean(img)
        std = torch.std(img, correction=1)
        background_value = torch.rand_like(img) * (2 * std) + (mean - std)

    else:
        raise ValueError("Please indicate a proper background imputation method!")

    img_distorted = mask * img + (1-mask) * background_value

    #Check if I need to restandardize the distorted image
    return img_distorted


def create_wavelet_distortion(wavelet, randomized_mask, background, device="cpu"):
    """
    Creates the distortion for the CartoonX algorithm
    :param mask:
    :param percent:
    :return: distorted image
    """

    if background == "gaussian":
        mean = torch.mean(wavelet)
        std = torch.std(wavelet, correction=1)
        background_value = std * torch.randn(wavelet.shape, device=device) + mean

    elif background == "uniform":
        mean = torch.mean(wavelet)
        std = torch.std(wavelet, correction=1)
        background_value = torch.rand(wavelet.shape, device=device) * (2 * std) + (mean - std)

    wavelet_distorted = wavelet * randomized_mask + (1-randomized_mask) * background_value

    return wavelet_distorted


def create_shearlet_distortion(shearlet, randomized_mask, background, device="cpu"):
    """
    Creates the distortion for the CartoonX algorithm
    :param mask:
    :param percent:
    :return: distorted image
    """

    if background == "gaussian":
        mean = torch.mean(shearlet)
        std = torch.std(shearlet, correction=1)
        background_value = std * torch.randn(shearlet.shape, device=device) + mean

    elif background == "uniform":
        mean = torch.mean(shearlet)
        std = torch.std(shearlet, correction=1)
        background_value = torch.rand(shearlet.shape, device=device) * (2 * std) + (mean - std)

    shearlet_distorted = shearlet * randomized_mask + (1-randomized_mask) * background_value

    return shearlet_distorted


def randomize_latent_mask_coefficients(mask, percentage, randomized_baseline=False, randomize_ascending=False):

    """
    This functions randomizes either based on saliency score or completely random the latent coefficients
    (for both, shearlet and wavelet)
    :param mask:
    :param percentage:
    :param randomized_baseline:
    :param randomize_ascending:
    :return:
    """
    mask_reshaped = mask.flatten()
    top_k = int(len(mask_reshaped) * percentage)

    if randomized_baseline:
        if randomize_ascending:
            new_mask = torch.ones_like(mask_reshaped, dtype=torch.float32)
            # For 0.0 percentage we keep everything
            if percentage != 0.0:
                random_indices = torch.randperm(len(new_mask))[:top_k-1]
                new_mask[random_indices] = 0.0
            new_mask = new_mask.reshape(mask.shape)

        else:
            new_mask = torch.zeros_like(mask_reshaped, dtype=torch.float32)
            if percentage != 0.0:
                random_indices = torch.randperm(len(new_mask))[:top_k-1]
                new_mask[random_indices] = 1.0
            new_mask = new_mask.reshape(mask.shape)

    else:
        sorted_mask_indices = torch.argsort(mask_reshaped, descending=True)
        threshold = mask_reshaped[sorted_mask_indices[top_k - 1]] if percentage != 0.0 else mask_reshaped[sorted_mask_indices[0]]

        if randomize_ascending:
            new_mask = (mask <= threshold).to(torch.float32)
        else:
            new_mask = (mask >= threshold).to(torch.float32)

    return new_mask


def get_distorted_img_cartoonX(masks, img, percentage, cartoonX, background, randomized_baseline=False,
                               randomize_ascending=False, device="cpu"):
    """
    Generates the distorted cartoonX image
    :param masks: List containing the masks, where the first element corresponds to the yl coefficients, and the rest to yh
    :param img: original image
    :param model: cartoonX model
    :param percentage: percentage to keep
    :return: distorted image
    """
    #Get the wavelet coefficients
    yl, yh = cartoonX.forward_dwt(img.to(device))

    randomized_mask_yl = randomize_latent_mask_coefficients(masks[0], percentage, randomized_baseline, randomize_ascending)
    yl_distorted = create_wavelet_distortion(yl, randomized_mask_yl, background, device=device)

    yh_distorted = []
    for mask, yh_coeff in zip(masks[1], yh):
        randomized_mask_yh = randomize_latent_mask_coefficients(mask, percentage, randomized_baseline, randomize_ascending)
        yh_distorted.append(create_wavelet_distortion(yh_coeff, randomized_mask_yh, background, device=device))

    distorted_image = cartoonX.inverse_dwt((yl_distorted, yh_distorted)).cpu()

    return distorted_image.clamp_(0, 1)

def get_distorted_img_shearletX(masks, img, percentage, shearletX, background,
                                randomized_baseline=False, randomize_ascending=False, device="cpu"):
    """
    Generates the distorted cartoonX image
    :param masks: List containing the masks, where the first element corresponds to the yl coefficients, and the rest to yh
    :param img: original image
    :param model: cartoonX model
    :param percentage: percentage to keep
    :return: distorted image
    """

    shearlet_coeffs = []
    for i in range(img.size(1)):
        coeffs = torchsheardec2D(img[:, i, :, :], shearletX.torch_shearlets, device=device).permute(0, 3, 1, 2)
        shearlet_coeffs.append(coeffs)

    shearlet_coeffs = torch.cat(shearlet_coeffs, dim=0)
    # Get the randomized mask back
    randomized_mask = randomize_latent_mask_coefficients(masks, percentage, randomized_baseline, randomize_ascending)
    shearlet_distorted = create_shearlet_distortion(shearlet_coeffs, randomized_mask, background, device=device)

    distorted_image = torchshearrec2D(
        shearlet_distorted, shearletX.torch_shearlets, shearletX.dualFrameWeights, device=device
    ).clamp(0, 1).unsqueeze(0)

    return distorted_image


def random_baseline_saliency(img, percentage, background, randomize_ascending=False, device="cpu"):

    if not torch.is_tensor(img):
        img = torch.tensor(img)

    img = img.to(device)

    # Get a random saliency coefficient in between 0 and 1
    saliency = torch.rand(img.flatten().shape)
    top_k = int(len(saliency) * percentage)

    sorted_mask_indices = torch.argsort(saliency, descending=True)
    threshold = saliency[sorted_mask_indices[top_k - 1]] if percentage != 0.0 else saliency[
        sorted_mask_indices[0]]

    if randomize_ascending:
        new_mask = (saliency <= threshold).to(torch.float32)
    else:
        new_mask = (saliency >= threshold).to(torch.float32)

    new_mask = new_mask.reshape(img.shape).to(device)

    if background == "mean":
        background_value = torch.mean(img).to(device)

    elif background == "zero":
        background_value = torch.zeros_like(img).to(device)

    elif background == "ones":
        background_value = torch.ones_like(img)

    elif background == "gaussian":
        mean = torch.mean(img)
        std = torch.std(img, correction=1)
        background_value = std * torch.randn(img.shape, device=device) + mean

    elif background == "uniform":
        mean = torch.mean(img)
        std = torch.std(img, correction=1)
        background_value = torch.rand(img.shape, device=device) * (2 * std) + (mean - std)

    else:
        raise ValueError("Please indicate a proper background imputation method!")

    img_distorted = new_mask * img + (1-new_mask) * background_value

    #Check if I need to restandardize the distorted image
    return img_distorted.clamp_(0, 1)


def get_distortion_various_percentage_levels(saliency_explanation, img, model,
                                             explainer=None, percentage_list=None, background="gaussian",
                                             randomize_ascending=False, latent_explainer="cartoonX", device="cpu"):
    """
    Calculates the distorted values for various percentage levels
    :param saliency_explanation: explanation map for the image under consideration
    :param img: original img
    :param model: model to get the explanations for
    :param percentage_list: percentage levels to keep, i.e. 0.1 means 0.1 features are kept and 0.9 are randomized
    :return: list of distorted values at various percentage levels
    """

    distorted_values = []
    if percentage_list is None:
        percentage_list = [round(x, 2) for x in np.arange(0.0, 1.05, 0.05)]

    if explainer is not None:
        if latent_explainer == "cartoonX":
            for p in percentage_list:
                distorted_img = get_distorted_img_cartoonX(saliency_explanation, img, p, explainer, background,
                                                           randomized_baseline=False,
                                                           randomize_ascending=randomize_ascending,
                                                           device=device)
                distorted_values.append(get_distortion(model, img, distorted_img, device=device))

        else:
            for p in percentage_list:
                distorted_img = get_distorted_img_shearletX(saliency_explanation, img, p, explainer, background,
                                                            randomized_baseline=False,
                                                            randomize_ascending=randomize_ascending,
                                                            device=device)
                distorted_values.append(get_distortion(model, img, distorted_img, device=device))

    else:
        for p in percentage_list:
            distorted_img = randomize_relevant_components(saliency_explanation, img, p, background,
                                                          randomize_ascending=randomize_ascending)
            distorted_values.append(get_distortion(model, img, distorted_img, device=device))

    return distorted_values


def get_distorted_random_baselines(img, model, percentage_list=None,
                                   background="uniform", randomize_ascending=False, device="cpu"):

    distorted_values = []
    if percentage_list is None:
        percentage_list = [round(x, 2) for x in np.arange(0.0, 1.05, 0.05)]

    for p in percentage_list:
        distorted_img = random_baseline_saliency(img, p, background, randomize_ascending=randomize_ascending, device=device)
        distorted_values.append(get_distortion(model, img, distorted_img, device=device))

    # Some quick checks that the value behave in the expected way
    relative_tolerance = 1e-5
    absolute_tolerance = 1e-5
    expected_value = 0.

    value_to_check = distorted_values[0] if randomize_ascending else distorted_values[-1]

    assert math.isclose(value_to_check, expected_value, rel_tol=relative_tolerance, abs_tol=absolute_tolerance), \
        f"Distortion {value_to_check} is not almost equal to {expected_value}"

    return distorted_values


def get_ssim_two_images(masked_image_1, masked_image_2):
    """
    Function to calculate the structural similarity index between two masks from different algorithms to check
    how similar the derived masked explanations are
    :param masked_image_1: masked image from mask_based algorithm 1 (tensor)
    :param masked_image_2: masked image from mask_based algorithm 2 (tensor)
    :return: structural similarity index which has range [-1, 1].
             1 indicates identical, 0 no similarity, -1 anti-correlation
    """
    # Check if we have grayscale images
    if masked_image_1.size(0) == 1:
        masked_image_1_np = masked_image_1.cpu().detach().numpy().squeeze(0)
        masked_image_2_np = masked_image_2.cpu().detach().numpy().squeeze(0)
        data_range = masked_image_2_np.max() - masked_image_2_np.min()
        ssim_value = np.round(ssim(masked_image_1_np, masked_image_2_np, data_range=data_range),  4)

    else:
        # If we have color images, then we take the average per channel
        ssim_values_per_channel = []
        for channel in range(masked_image_1.size(0)):
            masked_image_1_np = masked_image_1[channel].cpu().detach().numpy()
            masked_image_2_np = masked_image_2[channel].cpu().detach().numpy()
            data_range = masked_image_2_np.max() - masked_image_2_np.min()
            ssim_values_per_channel.append(np.round(
                ssim(masked_image_1_np, masked_image_2_np, data_range=data_range),
                4))
        ssim_value = np.round(np.mean(ssim_values_per_channel), 4)
    # Documentation advises to always calculate manually data range
    return ssim_value


def get_ssim_for_all_methods(explanation_dictionary):
    """
    Calculates the pair-wise ssmi between different masked images for different explanation algorithms
    :param explanation_dictionary: dictionary with keys being XAI method: str, and items the respective masked image (tensor)
    :return: dictionary with ssmi for each pair of XAI methods
    """

    explanation_methods = list(explanation_dictionary.keys())
    overall_ssmi_results = {}
    for i, explanation_method_a in enumerate(explanation_methods):
        for explanation_method_b in explanation_methods[i+1: ]:
            masked_image_a = explanation_dictionary[explanation_method_a]
            masked_image_b = explanation_dictionary[explanation_method_b]
            ssmi_key_name = "ssmi_" + explanation_method_a + "_" + explanation_method_b
            overall_ssmi_results[ssmi_key_name] = (
                get_ssim_two_images(masked_image_a, masked_image_b)
            )
    return overall_ssmi_results