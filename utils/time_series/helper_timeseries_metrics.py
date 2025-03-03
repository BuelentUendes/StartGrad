import torch
from sklearn import metrics
from tint.metrics import accuracy, comprehensiveness, cross_entropy, sufficiency

# Numerical constant for stability
EPS = 1e-5


def min_max_scale_attr(attr):
    # Min-max scaling (they call it normalize, but it is min-max scaling)
    # is also used in the code of the reference papers:
    # Check: https://github.com/josephenguehard/time_interpret/blob/main/tint/metrics/white_box/base.py

    assert len(attr.shape) == 3, "We expect the attr values to be of dim batch, seq, feature_dim! "
    min_per_mask = torch.amin(attr, dim=(1, 2), keepdim=True)
    max_per_mask = torch.amax(attr, dim=(1, 2), keepdim=True)

    attr_min_max_scaled = (attr - min_per_mask) / (max_per_mask - min_per_mask + EPS)

    # Assertions to check the scaling
    assert torch.all(attr_min_max_scaled >= 0.), "Min-max scaled values should be >= 0."
    assert torch.all(attr_min_max_scaled <= 1.), "Min-max scaled values should be <= 1."

    return attr_min_max_scaled


def get_aup_and_aur(saliency_values, true_salient_values, min_max_scaling=True, hard_labels=True):
    # Convert inputs to tensors if they are not
    saliency_values = torch.tensor(saliency_values) if not isinstance(saliency_values, torch.Tensor) \
        else saliency_values
    true_salient_values = torch.tensor(true_salient_values) if not isinstance(true_salient_values, torch.Tensor) \
        else true_salient_values.cpu()

    saliency_values = saliency_values.detach().cpu()

    if min_max_scaling:
        saliency_values = min_max_scale_attr(saliency_values)

    # Convert to int if hard_labels is True
    if hard_labels:
        true_salient_values = true_salient_values.int()

    precision, recall, thresholds = metrics.precision_recall_curve(true_salient_values.flatten(),
                                                                   saliency_values.flatten())
    # the two tensors need to be of the same length, so we take all but the last threshold
    # See: https://github.com/JonathanCrabbe/Dynamask/blob/main/experiments/results/state/get_results.py
    aup = metrics.auc(thresholds, precision[:-1]) if len(thresholds) > 1 else 0.0
    aur = metrics.auc(thresholds, recall[:-1]) if len(thresholds) > 1 else 0.0

    return round(aup, 4), round(aur, 4)


def get_mask_information(saliency_values, true_salient_values, min_max_scaling=True):
    # Mask information as defined in the paper:
    # Explaining Time Series Predictions with Dynamic Masks
    # Code from: https://github.com/JonathanCrabbe/Dynamask/blob/main/utils/metrics.py#L58

    saliency_values = saliency_values.detach().cpu()

    if min_max_scaling:
        saliency_values = min_max_scale_attr(saliency_values)

    # Select the subset of saliency_values that correspond to the true salient features as described in:
    # Explaining Time Series Predictions with Dynamic Masks

    saliency_values_subset = saliency_values[true_salient_values.cpu() == 1]

    mask_information = - torch.log2(1 - saliency_values_subset + EPS).sum()

    # We scale the results for better visualization
    mask_information /= 10**(4)

    return round(mask_information.item(), 4)


def get_mask_entropy(saliency_values, true_salient_values, min_max_scaling=True):
    # Mask entropy metric as defined in the paper:
    # Explaining Time Series Predictions with Dynamic Masks
    # Code from: https://github.com/JonathanCrabbe/Dynamask/blob/main/utils/metrics.py#L58

    saliency_values = saliency_values.detach().cpu()

    if min_max_scaling:
        saliency_values = min_max_scale_attr(saliency_values)

    # Select the subset of saliency_values that correspond to the true salient features as described in:
    # Explaining Time Series Predictions with Dynamic Masks
    saliency_values_subset = saliency_values[true_salient_values.cpu() == 1]

    mask_entropy = - (saliency_values_subset * torch.log2(saliency_values_subset + EPS)
                      + (1 - saliency_values_subset) * torch.log2(1 - saliency_values_subset + EPS))

    #Scale the results as done in the paper
    mask_entropy /= 10**3

    return round(mask_entropy.sum().item(), 4)


def get_black_box_metric_scores(model, x_test, mask, topk=0.2, baseline="zero"):

    baseline = torch.zeros_like(x_test) if baseline == "zero" else torch.mean(x_test, dim=1, keepdim=True)
    model.eval()
    black_box_metric_results = {}

    black_box_metric_results["Accuracy"] = accuracy(
        model,
        x_test,
        attributions=mask.cpu(),
        baselines=baseline,
        topk=topk,
    )
    black_box_metric_results["Comprehensiveness"] = comprehensiveness(
        model,
        x_test,
        attributions=mask.cpu(),
        baselines=baseline,
        topk=topk,
    )
    black_box_metric_results["CE"] = cross_entropy(
        model,
        x_test,
        attributions=mask.cpu(),
        baselines=baseline,
        topk=topk,
    )

    black_box_metric_results["Sufficiency"] = sufficiency(
        model,
        x_test,
        attributions=mask.cpu(),
        baselines=baseline,
        topk=topk,
    )

    return black_box_metric_results