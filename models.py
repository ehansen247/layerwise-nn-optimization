import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import helper

import config

class LayerConfigurableNN(nn.Module):
    '''
    Layer-wise configurable NN
    '''

    def __init__(self, hidden_layer_dim, etf=None):
        super().__init__()

        # Retrieve model configuration
        self.config = config.get_model_configuration()
        self.width, self.height, self.channels = self.config.get(
            "width"), self.config.get("height"), self.config.get("channels")
        self.batch_norm = self.config.get('batch_norm')
        self.num_classes = self.config.get("num_classes")

        self.input_block = self.get_input_layers()  # returns nn.Module
        if etf == None:
            self.simplex_ETF = helper.generate_simplex_etf(hidden_layer_dim, self.num_classes)
        self.hidden_blocks = []  # list of nn.Module

    def get_input_layers(self):
        raise NotImplementedError

    def output_transform(self):
        raise NotImplementedError

    def add_hidden_block(self, device):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def apply_ETF(self, x):
        return torch.matmul(x, self.simplex_ETF)

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
        x_output = sum(self.apply_ETF(self.output_transform(_x)) for _x in [x_inp] + x_hidden)

        return x_output

    def num_weights(self):
        sm = 0
        sm += sum(p.numel() for p in self.input_block.parameters())
        for block in self.hidden_blocks:
            sm += sum(p.numel() for p in block.parameters())

        return sm

    def num_trainable_weights(self):
        sm = 0
        sm += sum(p.numel() for p in self.input_block.parameters() if p.requires_grad)
        for block in self.hidden_blocks:
            sm += sum(p.numel() for p in block.parameters() if p.requires_grad)

        return sm

    def gradient_norm(self):
        total_norm = 0
        parameters = [p for p in self.input_block.parameters(
        ) if p.grad is not None and p.requires_grad]
        for block in self.hidden_blocks:
            parameters.extend([p for p in block.parameters()
                              if p.grad is not None and p.requires_grad])

        for p in parameters:
            grad_norm = p.grad.detach().data.norm(2)
            total_norm += grad_norm.item() ** 2
        total_norm = total_norm ** 0.5

        return total_norm

    def network_norm(self):
        total_norm = 0
        parameters = [p for p in self.input_block.parameters()]
        for block in self.hidden_blocks:
            parameters.extend([p for p in block.parameters()])

        for p in parameters:
            grad_norm = p.grad.detach().data.norm(2)
            total_norm += grad_norm.item() ** 2
        total_norm = total_norm ** 0.5

        return total_norm

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
    def __init__(self):
        self.hidden_layer_dim = config.get_model_configuration().get('hidden_layer_dim')

        super().__init__(self.hidden_layer_dim)

    def get_input_layers(self):
        inp_layers = []
        inp_layers.append(('flatten', nn.Flatten()))
        inp_dim = self.width * self.height * self.channels
        if self.batch_norm:
            inp_layers.append(('batch_norm', nn.BatchNorm1d(inp_dim)))
        inp_layers.extend([
            ("linear", nn.Linear(inp_dim,
                                 self.hidden_layer_dim)),
            ("inp_relu", nn.ReLU())
        ])

        return nn.Sequential(OrderedDict(inp_layers))

    def output_transform(self, x):
        return x

    def add_hidden_block(self, device):
        mlp_block = LayerwiseMLPBlock(hidden_layer_dim=self.hidden_layer_dim)
        self.hidden_blocks.append(mlp_block.to(device))

    def get_name(self):
        return "MLP"


class LayerwiseCNNBlock(nn.Module):
    def __init__(self, out_channels,
                 hidden_kernel_size, pool=False):
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
    def __init__(self, out_channels=12, init_kernel_size=3,
                 hidden_kernel_size=3, etf=None):
        self.out_channels = out_channels
        self.init_kernel_size = init_kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.mp_layers = 1
        self.padding = 1

        cfg = config.get_model_configuration()
        eff_width = cfg.get('width') + 2 * self.padding
        eff_height = cfg.get('height') + 2 * self.padding
        self.flatten_out_shape = int(
            self.out_channels *
            ((eff_width - self.init_kernel_size + 1) / (2 * self.mp_layers)) *
            ((eff_height - self.init_kernel_size + 1) / (2 * self.mp_layers))
        )

        super().__init__(self.flatten_out_shape, etf)

    def output_transform(self, x):
        return torch.flatten(x)

    def get_input_layers(self):
        inp_layers = []
        if self.batch_norm:
            inp_layers.append(('batch_norm', nn.BatchNorm2d(self.channels)))
        inp_layers.extend([
            ("input0", nn.Conv2d(self.channels, self.out_channels, self.init_kernel_size,
                                 padding=self.padding)),
            ("input1", nn.MaxPool2d(2)),
            ("input2", nn.ReLU())
        ])

        return nn.Sequential(OrderedDict(inp_layers))

    def add_hidden_block(self, device):
        cnn_block = LayerwiseCNNBlock(out_channels=self.out_channels,
                                      hidden_kernel_size=self.hidden_kernel_size,
                                      pool=True)
        self.hidden_blocks.append(cnn_block.to(device))

    def get_name(self):
        return "CNN"
