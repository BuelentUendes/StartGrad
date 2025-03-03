# Code credits:
# https://github.com/sanatonek/time_series_explainability/blob/master/data_generator/simulated_l2x_switchstate.py

import torch
import numpy as np

from tqdm import tqdm
from utils.time_series.helper_signal import GaussianProcess


SIG_NUM = 3
STATE_NUM = 3
P_S0 = [1 / 3]

imp_feature = [[0], [1], [2]]
correlated_feature = {0: {0: [1]}, 1: {1: [2]}, 2: {0: [1, 2]}}
scale = {0: [0.8, -0.5, -0.2], \
         1: [0, -1.0, 0], \
         2: [-0.2, -0.2, 0.8]}

transition_matrix = [[0.95, 0.02, 0.03],
                     [0.02, 0.95, 0.03],
                     [0.03, 0.02, 0.95]]


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = STATE_NUM
    cov = np.eye(SIG_NUM) * 0.1
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        covariance.append(c)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        m = scale[i]
        mean.append(m)
    mean = np.array(mean)
    return mean, covariance


def next_state(previous_state, t):
    p_vec = transition_matrix[previous_state]
    next_st = np.random.choice([0, 1, 2], p=p_vec)
    return next_st


def state_decoder(previous, next_st):
    return int(next_st * (1 - previous) + (1 - next_st) * previous)


def generate_linear_labels(X):
    logit = np.exp(-3 * np.sum(X, axis=1))

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_XOR_labels(X):
    y = np.exp(X[:, 0] * X[:, 1])

    prob_1 = np.expand_dims(1 / (1 + y), 1)
    prob_0 = np.expand_dims(y / (1 + y), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:, :4] ** 2, axis=1) - 1.5)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_additive_labels(X):
    logit = np.exp(-10 * np.sin(-0.2 * X[:, 0]) + 0.5 * X[:, 1] + X[:, 2] + np.exp(X[:, 3]) - 0.8)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def generate_linear_labels_v2(X):
    logit = np.exp(-0.2 * X[:, 0] + 0.5 * X[:, 1] + X[:, 2] + X[:, 3] - 0.8)

    prob_1 = np.expand_dims(1 / (1 + logit), 1)
    prob_0 = np.expand_dims(logit / (1 + logit), 1)

    y = np.concatenate((prob_0, prob_1), axis=1)

    return y


def create_signal(sig_len, gp_params, mean, cov):
    signal = None
    state_local = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    previous_label = None
    delta_state = 1

    # Sample for "previous" state (this is current state now)
    imp_sig = np.zeros(SIG_NUM)
    imp_sig[imp_feature[previous]] = 1
    importance.append(imp_sig)
    state_local.append(previous)

    for ii in range(1, sig_len):
        next_st = next_state(previous, delta_state)
        state_n = next_st

        imp_sig = np.zeros(SIG_NUM)
        if previous != state_n:
            # this samples labels+samples until  current point - before state change at next time point
            gp_vec = [GaussianProcess(lengthscale=g, mean=m, variance=0.1) for g, m in
                      zip(gp_params, mean[previous])]
            sample_ii = np.array([gp.sample_vectorized(time_vector=np.array(range(delta_state))) for gp in gp_vec])

            if signal is not None:
                signal = np.hstack((signal, sample_ii))
            else:
                signal = sample_ii

            y_probs = generate_linear_labels(sample_ii.T[:, imp_feature[previous]])

            y_logit = [yy[1] for yy in y_probs]
            y_label = [np.random.binomial(1, yy) for yy in y_logit]

            y.extend(y_label)
            y_logits.extend(y_logit)
            delta_state = 1
            imp_sig[imp_feature[state_n]] = 1
            imp_sig[-1] = 1
        else:
            delta_state += 1
        importance.append(imp_sig)

        # previous_label = y_label
        state_local.append(state_n)
        previous = state_n

    # sample points in the last state-change
    gp_vec = [GaussianProcess(lengthscale=g, mean=m, variance=0.1) for g, m in
              zip(gp_params, mean[previous])]
    sample_ii = np.array([gp.sample_vectorized(time_vector=np.array(range(delta_state))) for gp in gp_vec])

    # sometimes only one state is ever sampled
    if signal is not None:
        signal = np.hstack((signal, sample_ii))
    else:
        signal = sample_ii

    y_probs = generate_linear_labels(sample_ii.T[:, imp_feature[previous]])

    y_logit = [yy[1] for yy in y_probs]
    y_label = [np.random.binomial(1, yy) for yy in y_logit]

    y.extend(y_label)
    y_logits.extend(y_logit)

    # signal = signal
    y = np.array(y)
    importance = np.array(importance)

    return signal, y, state_local, importance, y_logits


def decay(x):
    return [0.9 * (1 - 0.1) ** x, 0.9 * (1 - 0.1) ** x]


def logit(x):
    return 1. / (1 + np.exp(-2 * (x)))


def normalize(train_data, test_data, config='mean_normalized'):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    len_of_stay = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == 'mean_normalized':
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in train_data])
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in test_data])
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


def create_switch_dataset(signal_len, sample_size):
    dataset = []
    labels = []
    importance_score = []
    states = []
    label_logits = []
    mean, cov = init_distribution_params()
    gp_lengthscale = np.random.uniform(0.2, 0.2, SIG_NUM)
    for _ in tqdm(range(sample_size), desc="Generate switch samples"):
        sig, y, state, importance, y_logits = create_signal(signal_len, gp_params=gp_lengthscale, mean=mean, cov=cov)
        dataset.append(sig.T)
        labels.append(y)
        importance_score.append(importance)
        states.append(state)
        label_logits.append(y_logits)

    x = torch.tensor(np.array(dataset), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)

    # Code from:
    # https://github.com/zichuan-liu/ContraLSP/blob/main/switchstate/switchloader.py
    true_saliency = torch.zeros(x.shape)
    for exp_id, time_slice in enumerate(states):
        for t_id, feature_id in enumerate(time_slice):
            true_saliency[exp_id, t_id, feature_id] = 1

    true_saliency = true_saliency.long()

    #importance_score = torch.tensor(np.array(importance_score), dtype=torch.float32)

    return x, labels, true_saliency



