import models
import config
import helpers

def get_top1_pos(outputs, targets):
    pred = np.argmax(outputs, axis=1)
    assert(len(pred) == len(targets))

    return np.sum(np.where(pred == targets, 1, 0))

def get_top5_pos(outputs, targets):
    sm = 0
    for i in range(len(targets)):
        top_5 = np.argpartition(outputs[i], -5)[-5:]
        sm += 1 if targets[i] in set(top_5) else 0

    return sm

def evaluate_model(model, data_loader, loss_function, device='cpu'):
    output_data = []
    targets_data = []
    current_loss = 0

    for i, data in enumerate(data_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Perform forward pass
        outputs = model(inputs)

        output_data.extend(outputs.cpu().detach().numpy())
        targets_data.extend(targets.cpu().detach().numpy())

        loss = loss_function(outputs, targets)
        current_loss += loss.item()

    N = len(targets_data)
    top1_acc = get_top1_pos(output_data, targets_data) / N
    top5_acc = get_top5_pos(output_data, targets_data) / N

    return current_loss, top1_acc, top5_acc


def train_model(model, epochs=None, train_acc=True, invariant=False, test_acc=True, debug=False):
    """ Train a model. """
    config = get_model_configuration()

    loss_function = config.get("loss_function")()
    optimizer = config.get("optimizer")(model.parameters(), lr=1e-4)
    trainloader = get_dataset(train=True, invariant=invariant)
    testloader = get_dataset(train=False, invariant=invariant)

#     Accelerate model
#     accelerator = accelerate.Accelerator()
#     model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

    # Iterate over the number of epochs
    entries = []

    if epochs is None:
        epochs = config.get("num_epochs")

    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        output_data = []
        targets_data = []

        # Iterate over the DataLoader for training data
        st_time = time.time()
        for i, data in enumerate(trainloader, 0):
            #             print(i)

            # Get inputs
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            current_loss += loss.item()

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

        end_time = time.time()

        if epoch % 5 == 0 or epoch == (epochs - 1):
            test_loss, test_top1_acc, test_top5_acc = evaluate_model(
                model, testloader, loss_function)
            train_loss, train_top1_acc, train_top5_acc = evaluate_model(
                model, trainloader, loss_function)
            print(f'Train Acc: {train_top1_acc}')
            print(f'Test Acc: {test_top1_acc}')
        else:
            test_loss, test_top1_acc, test_top5_acc = pd.NA, pd.NA, pd.NA
            train_loss, train_top1_acc, train_top5_acc = pd.NA, pd.NA, pd.NA

        elapsed_time = round(end_time - st_time, 1)
        train_entry = {'type': 'train', 'epoch': epoch, 'top1': train_top1_acc, 'top5': train_top5_acc,
                       'loss': current_loss, 'time': elapsed_time}

        print(f'Loss: {current_loss}')
        print(f'Time: {elapsed_time}')

        test_entry = {'type': 'test', 'epoch': epoch, 'top1': test_top1_acc, 'top5': test_top5_acc,
                      'loss': test_loss, 'time': pd.NA}

        entries.extend([train_entry, test_entry])

    # Return trained model
    return model, pd.DataFrame(entries), current_loss
