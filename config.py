import torch
from torch import nn

def get_global_configuration():
    """ Retrieve configuration of the training process. """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    global_config = {
        "config_batch": 0,
        "num_layers_to_add": 4,
        "rounds": 4, 
        "device": device,
        'invariant': False
    }

    return global_config

def get_model_configuration():
    """ Retrieve configuration for the model. """

    model_config = {
        "width": 32,
        "height": 32,
        "channels": 3,
        "num_classes": 10,
        "batch_size": 250,
        "loss_function": nn.CrossEntropyLoss,
        "optimizer": torch.optim.AdamW,
        "weight_decay": 0.01,
        "learning_rate": 0.001,
        "num_epochs": 3,
        "batch_norm": False,
        "hidden_layer_dim": 1024,
    }


    return model_config


# Start Learning Rate at 0.05, dropping it down choose three intervals and drop it twice
# SGD with learning rate scheduling

# Show how performance changes with one layer, next layer, next layer, etc.

# Report that observation of epochs vs. rounds in training

# Choose different sizes
# 512 for CIFAR-10, MNIST - 256
# Try 1024 - 
