def get_global_configuration():
    """ Retrieve configuration of the training process. """

    global_config = {
      "num_layers_to_add": 5,
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
      "optimizer": torch.optim.Adam,
      "num_epochs": 3,
      "hidden_layer_dim": 256,
    }

    return model_config
