import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict

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
        self.width, self.height, self.channels = self.config.get(
            "width"), self.config.get("height"), self.config.get("channels")
        self.num_classes = self.config.get("num_classes")

        #
        self.input_block = self.get_input_layers()  # returns nn.Module
        self.output_block = self.get_output_layers()  # returns nn.Module
        self.hidden_blocks = []  # list of nn.Module

    def get_input_layers(self):
        raise NotImplementedError

    def get_output_layers(self):
        raise NotImplementedError

    def add_hidden_block(self, device):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def forward(self, x):
        # print(x.shape)
        x_inp = self.input_block(x)
        # print(x_inp.shape)

        x_hidden = []
        x_next = x_inp
        for block in self.hidden_blocks:
            x_next = block(x_next)
            # print(x_next.shape)
            x_hidden.append(x_next)

        # print(sum(x_hidden).shape)
        x_output = self.output_block(x_next + x_inp + sum(x_hidden))
        # print(x_output.shape)

        return x_output

    def num_weights(self):
        sm = 0 
        sm += sum(p.numel() for p in self.input_block.parameters())
        sm += sum(p.numel() for p in self.output_block.parameters())
        for block in self.hidden_blocks:
            sm += sum(p.numel() for p in block.parameters())
            
        return sm

    def num_trainable_weights(self):
        sm = 0 
        sm += sum(p.numel() for p in self.input_block.parameters() if p.requires_grad)
        sm += sum(p.numel() for p in self.output_block.parameters() if p.requires_grad)
        for block in self.hidden_blocks:
            sm += sum(p.numel() for p in block.parameters() if p.requires_grad)
            
        return sm

    # def find_layer(self, layer):
    #     if layer == 'input':
    #         return self.input_block
    #     elif layer == 'output':
    #         return self.output_block
    #     else:
    #         assert(layer >= 0 and layer < len(self.hidden_blocks))
    #         return self.hidden_blocks[layer]

    def freeze_layer(self, layer: nn.Module):
        for param in layer.parameters():
            param.requires_grad = False

    def activate_layer(self, layer: nn.Module):
        for param in layer.parameters():
            param.requires_grad = True

    def freeze_layers(self, layers):
        for layer in layers:
            self.freeze_layer(layer)

    def activate_layers(self, layers):
        for layer in layers:
            self.activate_layer(layer)


class LayerwiseMLPBlock(nn.Module):
    def __init__(self, hidden_layer_dim=256):

        super().__init__()
        self.hidden_layer_dim = hidden_layer_dim

        self.layers = nn.Sequential(OrderedDict([
            ("block0", nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)),
            ("block1", nn.ReLU())
        ]))

    def forward(self, x):
        return self.layers(x)

class LayerwiseConfigurableMLP(LayerConfigurableNN):
    def __init__(self, hidden_layer_dim=256):
        self.hidden_layer_dim = hidden_layer_dim

        super().__init__()

    def get_input_layers(self):
        return nn.Sequential(OrderedDict([
            ("input0", nn.Flatten()),
            ("input1", nn.Linear(self.width * self.height * self.channels,
                                 self.hidden_layer_dim)),
            ("input2", nn.ReLU())
        ]))

    def get_output_layers(self):
        return nn.Sequential(OrderedDict([
            ("input0", nn.Linear(self.hidden_layer_dim, self.num_classes))
        ]))

    def add_hidden_block(self, device):
        mlp_block = LayerwiseMLPBlock(hidden_layer_dim=self.hidden_layer_dim)
        self.hidden_blocks.append(mlp_block.to(device))

    def get_name(self):
        return "MLP"


class LayerwiseCNNBlock(nn.Module):
    def __init__(self, out_channels,
                 hidden_kernel_size):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_kernel_size = hidden_kernel_size

        self.layers = nn.Sequential(OrderedDict([
            ("block0", nn.Conv2d(self.out_channels, self.out_channels, self.hidden_kernel_size,
                                 padding=1)),
            ("block1", nn.ReLU())
        ]))

    def forward(self, x):
        return self.layers(x)


class LayerwiseConfigurableCNN(LayerConfigurableNN):
    def __init__(self, out_channels=6, init_kernel_size=5,
                 hidden_kernel_size=3):
        self.out_channels = out_channels
        self.init_kernel_size = init_kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.mp_layers = 1
        self.padding = 1

        super().__init__()

    def get_output_layers(self):
        eff_width = self.width + 2 * self.padding
        eff_height = self.width + 2 * self.padding

        flatten_out_shape = int(
            self.out_channels *
            ((eff_width - self.init_kernel_size + 1) / (2 * self.mp_layers)) *
            ((eff_height - self.init_kernel_size + 1) / (2 * self.mp_layers))
        )

        return nn.Sequential(OrderedDict([
            ("output0", nn.Flatten()),
            ("output1", nn.Linear(flatten_out_shape, self.num_classes))
        ]))

    def get_input_layers(self):
        return nn.Sequential(OrderedDict([
            ("input0", nn.Conv2d(self.channels, self.out_channels, self.init_kernel_size,
                                 padding=self.padding)),
            ("input1", nn.MaxPool2d(2)),
            ("input2", nn.ReLU())
        ]))

    def add_hidden_block(self, device):
        cnn_block = LayerwiseCNNBlock(out_channels=self.out_channels,
                                      hidden_kernel_size=self.hidden_kernel_size)
        self.hidden_blocks.append(cnn_block.to(device))

    def get_name(self):
        return "CNN"
