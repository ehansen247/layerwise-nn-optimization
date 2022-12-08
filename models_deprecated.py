class LayerConfigurableNN(nn.Module):
    '''
    Layer-wise configurable NN
    '''

    def __init__(self, added_layers=0):
        super().__init__()

        # Retrieve model configuration
        self.config = get_model_configuration()
        self.width, self.height, self.channels = config.get(
            "width"), config.get("height"), config.get("channels")
        self.flatten_shape = config.get("width") * config.get("height") * config.get("channels")
        self.layer_dim = config.get("layer_dim")
        self.num_classes = config.get("num_classes")

        # Create layer structure
        layers = self.init_layers()

        for i in range(added_layers):
            self.add_layer()

        # Create output layers
        for layer in self.get_output_layers():
            layers.append((str(len(layers)), layer))

        # Initialize the Sequential structure
        self.layers = nn.Sequential(OrderedDict(layers))

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

    def add_layer(self):
        """ Add a new layer to a model, setting all others to nontrainable. """
        config = get_model_configuration()

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
