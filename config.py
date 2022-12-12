import torch
from torch import nn

from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST


def get_global_configuration():
    """ Retrieve configuration of the training process. """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_config = {
        "config_batch": 0,
        "num_layers": 3,
        "rounds": 5,
        "device": device,
        'invariant': False,
        'dataset': MNIST,
        "condition": "exp2_rounds5_epochs20_MNIST"
    }

    return global_config

def get_model_configuration():
    """ Retrieve configuration for the model. """

    model_config = {
        "width": 28,
        "height": 28,
        "channels": 1,
        "num_classes": 10,
        "batch_size": 250,
        "loss_function": nn.CrossEntropyLoss,
        "optimizer": torch.optim.AdamW,
        "weight_decay": 0.01,
        "learning_rate": 0.001,
        "num_epochs": 3,
        "batch_norm": True,
        "hidden_layer_dim": 128,
    }

    return model_config


# Start Learning Rate at 0.05, dropping it down choose three intervals and drop it twice
# SGD with learning rate scheduling

# Show how performance changes with one layer, next layer, next layer, etc.

# Report that observation of epochs vs. rounds in training

# Choose different sizes
# 512 for CIFAR-10, MNIST - 256
# Try 1024 -
