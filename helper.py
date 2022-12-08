import config
import numpy as np
import torch
import os

def get_dataset(train=True, invariant=False):
    """ Load and convert dataset into inputs and targets """
    config = config.get_model_configuration()
    if invariant:
        T = transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()]),
            transforms.RandomRotation((0, 360)),
            transforms.ToTensor(),
        ])
    else:
        T = transforms.ToTensor()
    dataset = CIFAR10(os.getcwd(), train=train, download=True, transform=T)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.get("batch_size"), shuffle=True, num_workers=1)

    return trainloader

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
