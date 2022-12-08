import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from accelerate import Accelerator

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config

class LayerConfigurableNN(nn.Module):
    '''
    Layer-wise configurable NN
    '''

    def __init__(self):
        super().__init__()

        # Retrieve model configuration
        self.config = config.get_model_configuration()
        self.width, self.height, self.channels = config.get(
            "width"), config.get("height"), config.get("channels")
        self.num_classes = config.get("num_classes")

        #
        self.input_block = self.get_input_layers()  # returns nn.Module
        self.output_block = self.get_output_layers()  # returns nn.Module
        self.hidden_blocks = []  # list of nn.Module

    def get_input_layers(self):
        raise NotImplementedError

    def get_output_layers(self):
        raise NotImplementedError

    def add_hidden_block(self):
        raise NotImplementedError

    def forward(self, x):
        x_inp = self.input_block(x)

        x_hidden = []
        x_next = x_inp
        for block in self.hidden_blocks:
            x_next = block(x_next)
            x_hidden.append(x_next)

        x_output = self.output_block(x_next + x_inp + sum(x_hidden))

        return x_output

    def num_weights(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def find_layer(self, layer)
        for layer in layers:
            if layer == 'input':
                return self.input_block
            elif layer == 'output':
                return self.output_block
            else:
                assert(layer >= 0 and layer < len(self.hidden_blocks))
                return self.hidden_blocks[layer]

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def activate_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def freeze_layers(layers, status):
        for layer in layers:
            self.freeze_layer(self.find_layer(layer))

    def activate_layers(self, layers):
        for layer in layers:
            self.activate_layer(self.find_layer(layer))


class LayerwiseMLPBlock(nn.Module):
    def __init__(self, hidden_layer_dim=256):
      self.layers = nn.Sequential(
          ("block0", nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)),
          ("block1": nn.ReLU())
      )

    def forward(self, x):
        return self.layers(x)

class LayerwiseConfigurableMLP(LayerConfigurableNN):
    def __init__(self, hidden_layer_dim=256):
        self.hidden_layer_dim = hidden_layer_dim

        super().__init__()

    def get_input_layers(self):
        return nn.Sequential([
            ("input0", nn.Flatten()),
            ("input0", nn.Linear(self.width * self.height * self.channels,
              self.hidden_layer_dim)),
            ("input1"), nn.ReLU())
        ])

    def get_output_layers(self):
        return nn.Sequential([
            ("input0", nn.Linear(self.hidden_layer_dim, self.num_classes))
        ])

    def add_hidden_block(self):
        self.hidden_blocks.append(
          LayerwiseMLPBlock(hidden_layer_dim=self.hidden_layer_dim)
        )


class LayerwiseCNNBlock(nn.Module):
    def __init__(self, out_channels,
                 hidden_kernel_size):

    self.layers=nn.Sequential(
        ("block0", nn.Conv2d(self.out_channels, self.out_channels, self.hidden_kernel_size)),
        ("block1": nn.ReLU())
    )

    def forward(self, x):
        return self.layers(x)


class LayerwiseConfigurableCNN(LayerConfigurableNN):
    def __init__(self, out_channels = 6, init_kernel_size = 8,
                 self.hidden_kernel_size = 4):
        self.out_channels=out_channels
        self.init_kernel_size=init_kernel_size
        self.hidden_kernel_size=hidden_kernel_size

        super().__init__()

    def get_output_layers(self):
        flatten_out_shape=int(
            self.out_channels * (self.width / (2 * self.mp_layers) - self.hidden_kernel_size)
            * (self.height / (2 * self.mp_layers) - self.hidden_kernel_size)
        )

        return nn.Sequential([
            ("output0", nn.Flatten(),
            ("output1", nn.Linear(self.flatten_out_shape, self.num_classes)
        ])

    def get_input_layers(self):
        return nn.Sequential([
          ("input0", nn.Conv2d(self.channels, self.out_channels, self.init_kernel_size)),
          ("input1", nn.MaxPool2d(2)),
          ("input2", nn.ReLU())
        ])

    def add_hidden_block(self):
        self.hidden_blocks.append(
          LayerwiseCNNBlock(out_channels=self.out_channels,
                            hidden_kernel_size=self.hidden_kernel_size)
        )
