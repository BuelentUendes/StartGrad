# Code from:
# https://github.com/JonathanCrabbe/Dynamask/blob/main/fit/data_generator/state_data.py

from tqdm import tqdm
import torch
import numpy as np

SIG_NUM = 3
STATE_NUM = 1
P_S0 = [0.5]

correlated_feature = [0, 0]  # Features that re correlated with the important feature in each state

imp_feature = [1, 2]  # Feature that is always set as important
scale = [[0.1, 1.6, 0.5], [-0.1, -0.4, -1.5]]  # Scaling factor for distribution mean in each state
trans_mat = np.array([[0.1, 0.9], [0.1, 0.9]])


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = np.power(2, STATE_NUM)
    cov = np.eye(SIG_NUM) * 0.8
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        c[imp_feature[i], correlated_feature[i]] = 0.01
        c[correlated_feature[i], imp_feature[i]] = 0.01
        c = c + np.eye(SIG_NUM) * 1e-3
        covariance.append(c)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        m = scale[i]
        mean.append(m)
    mean = np.array(mean)
    return mean, covariance


def next_state(previous_state, t):
    if previous_state == 1:
        params = 0.95
    else:
        params = 0.05

    params = params - float(t / 500) if params > 0.8 else params
    next = np.random.binomial(1, params)
    return next


def state_decoder(previous, next):
    return int(next * (1 - previous) + (1 - next) * (previous))


def create_signal(sig_len, mean, cov):
    signal = []
    states = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    delta_state = 0
    for i in range(sig_len):
        next = next_state(previous, delta_state)
        state_n = next

        if state_n == previous:
            delta_state += 1
        else:
            delta_state = 0

        imp_sig = np.zeros(3)
        if state_n != previous or i == 0:
            imp_sig[imp_feature[state_n]] = 1

        importance.append(imp_sig)
        sample = np.random.multivariate_normal(mean[state_n], cov[state_n])
        previous = state_n
        signal.append(sample)
        y_logit = logit(sample[imp_feature[state_n]])
        y_label = np.random.binomial(1, y_logit)
        y.append(y_label)
        y_logits.append(y_logit)
        states.append(state_n)
    signal = np.array(signal)
    y = np.array(y)
    importance = np.array(importance)

    return signal.T, y, states, importance, y_logits


def decay(x):
    return [0.9 * (1 - 0.1) ** x, 0.9 * (1 - 0.1) ** x]


def logit(x):
    return 1.0 / (1 + np.exp(-2 * (x)))


def normalize(train_data, test_data, config="mean_normalized"):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    len_of_stay = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)

    if config == "mean_normalized":
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for x in train_data]
        )
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for x in test_data]
        )
    elif config == "zero_to_one":
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])

    return train_data_n, test_data_n


def create_state_dataset(signal_len, sample_size):
    dataset = []
    labels = []
    importance_score = []
    states = []
    label_logits = []
    mean, cov = init_distribution_params()

    for _ in tqdm(range(sample_size), desc="Generate state samples:"):
        sig, y, state, importance, y_logits = create_signal(signal_len, mean, cov)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance.T)
        states.append(state)
        label_logits.append(y_logits)

    dataset = np.array(dataset)
    labels = np.array(labels)

    observations_tensor = torch.permute(torch.tensor(np.array(dataset), dtype=torch.float32), (0, 2, 1))
    y_tensor = torch.tensor(np.array(labels), dtype=torch.float32)

    true_states = np.stack(states)
    true_states += 1

    true_saliency = torch.zeros(observations_tensor.shape)
    for exp_id, time_slice in enumerate(true_states):
        for t_id, feature_id in enumerate(time_slice):
            true_saliency[exp_id, t_id, feature_id] = 1
    true_saliency = true_saliency.long()

    return observations_tensor, y_tensor, true_saliency


def standardize_input(x, mean=None, std=None):
    """

    :param x: dimension Sample_size x T X D where D is dimensionality
    :param mean:
    :param std:
    :return:
    """

    if mean is None:
        feature_mean = x.mean(dim=(0, 1), keepdim=True)

    else:
        assert mean.shape == (1, 1, x.shape[-1])
        feature_mean = mean

    if std is None:
        feature_std = x.std(dim=(0, 1), keepdim=True)

    else:
        assert std.shape == (1, 1, x.shape[-1])
        feature_std = std

    x_standardized = (x - feature_mean) / feature_std

    return x_standardized, feature_mean, feature_std

