class LayerConfigurableNN(nn.Module):
    '''
    Layer-wise configurable NN
    '''

    def __init__(self, added_layers=0):
        super().__init__()

        # Retrieve model configuration
        self.config = config.get_model_configuration()
        self.width, self.height, self.channels = self.config.get(
            "width"), self.config.get("height"), self.config.get("channels")
        self.flatten_shape = self.width * self.height * self.channels
        print(self.channels)
        self.layer_dim = self.config.get("hidden_layer_dim")
        self.num_classes = self.config.get("num_classes")

        self.layer_component_map = {}
        self.num_layer_components = 0

        # Create layer structure
        init_layers = self.init_layers()

        for i in range(len(init_layers)):
            self.layer_component_map[i] = self.num_layer_components  # 0-indexed
        self.num_layer_components += 0

        # Create output layers
        for layer in self.get_output_layers():
            init_layers.append((str(len(init_layers)), layer))

        # Initialize the Sequential structure
        self.layers = nn.Sequential(OrderedDict(init_layers))

        for i in range(added_layers):
            self.add_layer()

#         self.to(device)

    # Note that layer_num is 0 indexed
    def activate_single_layer(self, layer_component_num):
        for i, layer in enumerate(self.layers):
            for param in old_layer.parameters():
                param.requires_grad = (self.layer_component_map[i] == layer_component_num)

    def init_layers(self):
        raise NotImplementedError

    def get_intermediate_layers(self):
        raise NotImplementedError

    def get_output_layer(self):
        raise NotImplementedError

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def set_structure(self, layers):
        self.layers = nn.Sequential(OrderedDict(layers))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_name(self):
        raise NotImplementedError

    def add_layer(self):
        """ Add a new layer to a model, setting all others to nontrainable. """
        config = config.get_model_configuration()

        # Retrieve current layers
        layers = self.layers
        print("=" * 50)
        print("Old structure:")
        print(layers)

        # Save last layer for adding later
        last_layer = layers[-1]

        # Define new structure
        new_structure = []

        # Iterate over all except last layer
        for layer_index in range(len(layers) - 1):

            # For old layer, set all parameters to nontrainable
            old_layer = layers[layer_index]
            for param in old_layer.parameters():
                param.requires_grad = False

            # Append old layer to new structure
            new_structure.append((str(layer_index), old_layer))

        # Append new layer to the final intermediate layer
        new_layers = self.get_intermediate_layers()
        for layer in new_layers:
            new_structure.append((str(len(new_structure)), layer))

        # Re-add last layer
        new_structure.append((str(len(new_structure)), last_layer))

        # Change the model structure
        self.set_structure(new_structure)

        # Return the model
        print("=" * 50)
        print("New structure:")
        print(self.layers)

# The images in CIFAR-10 are of size 3x32x32

class LayerConfigurableCNN(LayerConfigurableNN):
    '''
    Layer-wise configurable CNN.
    '''

    def __init__(self, added_layers=0):
        self.out_channels = 6
        self.init_kernel_size = 8
        self.hidden_kernel_size = 4
        self.mp_layers = 1  # max pool layers

        super().__init__()

    def get_name(self):
        return 'CNN'

    def init_layers(self):
        return [(str(0), nn.Conv2d(self.channels, self.out_channels, self.init_kernel_size)),
                (str(1), nn.MaxPool2d(2)),
                (str(2), nn.ReLU())]

    def get_intermediate_layers(self):
        return [nn.Conv2d(self.out_channels, self.out_channels, self.hidden_kernel_size), nn.ReLU()]

    def get_output_layers(self):
        self.flatten_out_shape = int(self.out_channels * (self.width / (2 * self.mp_layers) - self.hidden_kernel_size
                                                          ) * (self.height / (2 * self.mp_layers) - self.hidden_kernel_size))

        return [nn.Flatten(), nn.Linear(self.flatten_out_shape, self.num_classes)]


class LayerConfigurableMLP(LayerConfigurableNN):
    '''
    Layer-wise configurable Multilayer Perceptron.
    '''

    def __init__(self, added_layers=0):
        super().__init__()

    def init_layers(self):
        return [
            (str(0), nn.Flatten()),
            (str(1), nn.Linear(self.flatten_shape, self.layer_dim)),
            (str(2), nn.ReLU())
        ]

    def get_name(self):
        return 'MLP'

    def get_intermediate_layers(self):
        return [nn.Linear(self.layer_dim, self.layer_dim), nn.ReLU()]

    def get_output_layers(self):
        return [nn.Linear(self.layer_dim, self.num_classes)]
