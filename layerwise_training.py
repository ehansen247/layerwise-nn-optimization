import config
import models
import torch

def greedy_layerwise_training(model):
    """ Perform greedy layer-wise training. """

    print("NEW!")
    global_config = config.get_global_configuration()
    torch.manual_seed(42)

    # Loss comparison
    loss_comparable = float('inf')

    # Iterate over the number of layers to add
    training_losses = []
    top5_accs = []
    top1_accs = []

    dfs = []
    for num_layers in range(global_config.get("num_layers_to_add")):
        # Print which model is trained
        print("=" * 100)
        if num_layers > 0:
            print(f">>> TRAINING THE MODEL WITH {num_layers} ADDITIONAL LAYERS:")
        else:
            print(f">>> TRAINING THE BASE MODEL:")

        # Train the model
        model, df, end_loss = train_model(model, invariant=global_config['invariant'])
        df['layer'] = num_layers
        df['layer_params'] = model.num_trainable_weights()
        dfs.append(df)

        # Compare loss
        if num_layers > 0 and end_loss < loss_comparable:
            print("=" * 50)
            print(
                f">>> RESULTS: Adding this layer has improved the model loss from {loss_comparable} to {end_loss}")
            loss_comparable = end_loss
        elif num_layers > 0:
            print("=" * 50)
            print(
                f">>> RESULTS: Adding this layer did not improve the model loss from {loss_comparable} to {end_loss}")
        elif num_layers == 0:
            loss_comparable = end_loss

        # Add layer to model
        model.add_hidden_block()
        model = model.to(device)

    # Process is complete
    print("Training process has finished.")

    model_config = config.get_model_configuration()
    results_df = pd.concat(dfs)
    results_df['optimizer'] = str(model_config['optimizer'])
    results_df['hidden_layer_dim'] = model_config['hidden_layer_dim']
    results_df['batch_size'] = model_config['batch_size']
    results_df['model'] = model.get_name()
    results_df['invariant'] = global_config['invariant']

    return model, results_df
