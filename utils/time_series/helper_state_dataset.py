# Implementation of Timeseries XAI
# Code is based on:
# https://github.com/josephenguehard/time_interpret/blob/main/tint/datasets/hmm.py
# https://github.com/sanatonek/time_series_explainability/blob/master/data_generator/state_data.py
# _________________________________________________________________________________________________

import os

import numpy as np
import torch
from tqdm import tqdm
import pickle

from utils.general.helper_path import DATA_PATH

TARGET_PATH = os.path.join(DATA_PATH, "Time_Series")
STATE_NUM = 1
NUM_FEATURES = 3
SIGMA_DIAG = 0.8
STATE_MEANS = [[0.1, 1.6, 0.5], [-0.1, -0.4, -1.5]]
IMPORTANT_FEATURES_IDX = [1, 2]
CORRELATED_FEATURE = [0, 0]
INIT_PROBABILITY = 0.5


def create_directory(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def save_dataset(data, base_path, filename):
    save_name = os.path.join(base_path, filename) + ".pkl"
    with open(save_name, "wb") as f:
        pickle.dump(data, f)


def init_covariance_params(num_states, num_features,
                           sigma_diag=0.8,
                           imp_feature=None,
                           correlated_feature=None):
    # Code adjusted
    # from: https://github.com/sanatonek/time_series_explainability/blob/master/data_generator/state_data.py
    # Covariance matrix is constant across states but distribution means change based on the state value

    if imp_feature is None:
        imp_feature = [1, 2]

    if correlated_feature is None:
        correlated_feature = [0, 0]

    state_count = 2 ** num_states
    cov = np.eye(num_features)*sigma_diag
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        c[imp_feature[i], correlated_feature[i]] = 0.01
        c[correlated_feature[i], imp_feature[i]] = 0.01
        c = c + np.eye(num_features) * 1e-3
        covariance.append(c)
    covariance = np.array(covariance)
    return covariance


def generate_next_state(previous_state,
                        timestep,
                        trans_prob_state_1=0.95,
                        dynamic_transition=True,
                        decay=0.002,
                        ):

    if previous_state: # 1 is truthy
        params = trans_prob_state_1
    else:
        params = (1 - trans_prob_state_1)

    if dynamic_transition and previous_state:
        params -= timestep * decay

    # Ensure probability stays in valid range (0, 1)
    params = max(0, min(1, params))
    next_state = np.random.binomial(1, params)
    return next_state


def calculate_label_prob(observation, state, important_features, coefficient=2):
    """
    Calculates the label probability based on the state and the important features
    :param observation: D-dimensional observation
    :param state: binary state [0, 1]
    :param important_features: idx of the important feature
    :param coefficient: affects the logits, in the paper it was set to -2
    :return: probability of the next state
    """
    # When state 0 -> feature 2 is selected (in Python idx 1)
    # When state 1 -> feature 3 is selected (in Python idx 2)
    feature_indices = important_features[state]
    selected_features = observation[feature_indices]
    return np.asarray(1 / (1 + np.exp(-coefficient * selected_features)))


def create_time_series_signal(
        signal_length,
        state_means,
        state_covs,
        init_probability=0.5,
        important_features_idx=None,
):

    if important_features_idx is None:
        important_features_idx = [1, 2]

    previous_state = np.random.binomial(1, init_probability)
    observations = []
    states = []
    y_logits = []
    y = []
    # list of one-hot encoded, where 1 indicates the important feature and 0 otherwise
    important_features = []
    # State covs is shape state_numbers x features x features
    num_features = state_covs.shape[1]

    for t in range(signal_length):

        next_state = generate_next_state(
            previous_state,
            timestep=t,
            trans_prob_state_1=0.95,
            dynamic_transition=True,
            decay=0.002)

        # IMPORTANT NOTE: The task is to find the important time step and feature where the state transition occurs!
        # See paper: https://papers.nips.cc/paper_files/paper/2020/file/08fa43588c2571ade19bc0fa5936e028-Paper.pdf

        important_feature = [0] * num_features
        important_feature[important_features_idx[next_state]] = 1

        important_features.append(important_feature)

        observation = np.random.multivariate_normal(state_means[next_state], state_covs[next_state])
        y_logit = calculate_label_prob(observation, next_state, important_features_idx, coefficient=2)

        observations.append(observation)
        y_logits.append(y_logit)

        y_label = np.random.binomial(1, y_logit)
        y.append(y_label)
        states.append(next_state)

        previous_state = next_state

    return observations, y, y_logits, states, important_features


def create_state_dataset(
        signal_length,
        sample_size,
        init_probability=INIT_PROBABILITY,
        state_num=STATE_NUM,
        state_means=STATE_MEANS,
        num_features=NUM_FEATURES,
        sigma_diag=SIGMA_DIAG,
        important_features_idx=IMPORTANT_FEATURES_IDX,
        correlated_feature=CORRELATED_FEATURE,
):

    observations = []
    y = []
    y_logits = []
    states = []
    important_features = []

    for _ in tqdm(range(sample_size), desc="Generate samples:"):
        variance = init_covariance_params(state_num, num_features, sigma_diag, important_features_idx,
                                          correlated_feature)
        sample_observation, sample_y, sample_y_logits, sample_states, sample_important_features = create_time_series_signal(
            signal_length, state_means, variance, init_probability)

        observations.append(sample_observation)
        y.append(sample_y)
        y_logits.append(sample_y_logits)
        states.append(sample_states)
        important_features.append(sample_important_features)

    observations_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    important_features = torch.tensor(np.array(important_features), dtype=torch.float32)

    return observations_tensor, y_tensor, important_features


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
