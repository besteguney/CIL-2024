from .unet import UNet

import yaml
from pathlib import Path
import torch
import torch.nn as nn

CHECKPOINT_PATH = Path("checkpoints")
CONFIG_PATH = Path("configs/")


def save_config(config, file_name: str):
    with open(CONFIG_PATH / file_name, 'w') as file:
        yaml.dump(config.__dict__, file)


def load_config(file_name: str):
    file_name += ".yaml" if not file_name.endswith(".yaml") else ""
    with open(CONFIG_PATH / file_name, 'r') as file:
        config_dict = yaml.safe_load(file)

    if "model_type" in config_dict:
        return config_dict
    else:
        raise ValueError("model_type missing from config")


def model_from_config(config, device="cpu"):
    model_type = config["model_type"]
    if model_type == "UNet":
        model = UNet(config)
    else:
        raise ValueError(f"Invalid model_type, got {model_type}")
    
    model.to(device)
    return model


def loss_fn_from_config(config):
    loss_fn_name = config["loss_fn"]
    if loss_fn_name == "BCE":
        loss_fn = nn.BCELoss()
    else:
        raise ValueError(f"Invalid loss function, got {loss_fn_name}")
    return loss_fn


def model_from_checkpoint(checkpoint_path: str, device="cpu"):
    checkpoint = torch.load(CHECKPOINT_PATH / checkpoint_path, map_location=device)
    config = load_config(checkpoint["config_name"])
    model = model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model