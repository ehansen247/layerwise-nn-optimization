import models
import config
import helpers

def test_model(model, loss_function):
    testloader = get_dataset(train=False)

    output_data = []
    targets_data = []
    current_loss = 0

    for i, data in enumerate(testloader):
        inputs, targets = data

        # Perform forward pass
        outputs = model(inputs)

        output_data.extend(outputs.detach().numpy())
        targets_data.extend(targets.detach().numpy())

        loss = loss_function(outputs, targets)
        current_loss += loss.item()

    N = len(targets_data)
    top1_acc = get_top1_pos(output_data, targets_data) / N
    top5_acc = get_top5_pos(output_data, targets_data) / N

    return current_loss, top1_acc, top5_acc

def train_model(model, epochs=None, debug=False):
    """ Train a model. """
    config = get_model_configuration()
    loss_function = config.get("loss_function")()
    optimizer = config.get("optimizer")(model.parameters(), lr=1e-4)
    trainloader = get_dataset()
    accelerator = Accelerator()

    # Accelerate model
    model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

    # Iterate over the number of epochs
    entries = []

    if epochs is None:
        epochs = config.get("num_epochs")

    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Positive / Accuracy Rate
        top_1_positives = 0
        top_5_positives = 0
        n = 0

        output_data = []
        targets_data = []

        # Iterate over the DataLoader for training data
        st_time = time.time()
        for i, data in enumerate(trainloader, 0):
            #             print(i)

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
#             print(outputs)
#             print(outputs.shape)
#             print(targets)
#             print(targets.shape)
            loss = loss_function(outputs, targets)

#             o, t = outputs.detach().numpy(), targets.detach().numpy()
#             top_1_positives += get_top1_pos(o, t)
#             top_5_positives += get_top5_pos(o, t)
#             n += len(targets)

            output_data.extend(outputs.cpu().detach().numpy())
            targets_data.extend(targets.cpu().detach().numpy())
            current_loss += loss.item()

            # Perform backward pass
            accelerator.backward(loss)

            # Perform optimization
            optimizer.step()

            # Print statistics
            if debug:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
#             end_loss = current_loss / 500
#             current_loss = 0.0

        end_time = time.time()

        top1_acc = get_top1_pos(output_data, targets_data) / len(targets_data)
        top5_acc = get_top5_pos(output_data, targets_data) / len(targets_data)

        train_entry = {'type': 'train', 'epoch': epoch, 'top1': top1_acc, 'top5': top5_acc,
                       'loss': current_loss, 'time': round(end_time - st_time, 1)}

        test_st_time = time.time()
        test_loss, test_top1_acc, test_top5_acc = test_model(model, loss_function)
        test_end_time = time.time()

        print(f'Loss: {current_loss}')
        print(f'Train Acc: {top1_acc}')
        print(f'Test Acc: {test_top1_acc}')

        test_entry = {'type': 'test', 'epoch': epoch, 'top1': test_top1_acc, 'top5': test_top5_acc,
                      'loss': test_loss, 'time': round(test_st_time - test_end_time, 1)}

        entries.extend([train_entry, test_entry])

    print(n)
    print(top_1_positives)
    print(top_5_positives)

    # Return trained model
    return model, pd.DataFrame(entries), current_loss
