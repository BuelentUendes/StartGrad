# Helper file for used paths
import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = abspath(os.path.join(FILE_PATH, './../../'))
DATA_PATH = abspath(os.path.join(BASE_PATH, './', "data"))
CUSTOM_PATH = abspath(os.path.join(DATA_PATH, 'Custom'))
TIME_SERIES_PATH = abspath(os.path.join(DATA_PATH, "Time_Series"))
IMAGENET_PATH = abspath(os.path.join(DATA_PATH, 'ImageNet'))
IMAGENET_DATA_PATH = abspath(os.path.join(IMAGENET_PATH, 'validation_set'))
IMAGENET_TUNING_DATA_PATH = abspath(os.path.join(IMAGENET_PATH, 'hyperparameter_tuning'))
MODELS_PATH = abspath(os.path.join(BASE_PATH, 'saved_models'))
CONFIG_PATH = abspath(os.path.join(BASE_PATH, 'config_environment'))
FIGURES_PATH = abspath(os.path.join(BASE_PATH, 'figures'))
RESULTS_PATH = abspath(os.path.join(BASE_PATH, 'results'))
PERFORMANCE_TRADEOFF_PATH = abspath(os.path.join(BASE_PATH, 'results', "performance_tradeoff"))
PERFORMANCE_STARTGRAD_PATH = abspath(os.path.join(BASE_PATH, 'results', "performance_runtime_startgrad"))
