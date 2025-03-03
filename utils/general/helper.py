import random
import os
import yaml
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_yaml_config_file(path_to_yaml_file: str):
    try:
        with open(path_to_yaml_file) as file:
            config = yaml.safe_load(file)
        return config

    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file at {path_to_yaml_file} cannot be found!"
        )

    except yaml.YAMLError:
        raise yaml.YAMLError(
            "Error parsing the YAML file"
        )