import config
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms
import datetime
import pandas as pd

from scipy.stats import ortho_group

def generate_simplex_etf(in_dim, out_dim, seed=520):
    np.random.seed(seed)
    
    mat = np.random.uniform(-1, 1, size=(in_dim, out_dim))
    q, _ = np.linalg.qr(mat)
    
    M = np.sqrt(out_dim / (out_dim - 1)) * np.identity(out_dim) - \
        (1 / float(out_dim)) * np.ones((out_dim, out_dim))
    
    etf = np.matmul(q, M)
    
    tensor_etf = torch.from_numpy(etf.astype(np.float32))
    assert(tensor_etf.requires_grad == False)
    
    return tensor_etf

def get_datetime_str(dt):
    return str(dt.strftime("%Y-%m-%d-%H-%M"))

def get_results_files(results_dir, st_file, end_file):
    dfs = []
    files = os.listdir(results_dir)
    for file in files:
        filename = file.split('.')[0]
        if filename >= st_file and filename <= end_file:
            dfs.append(pd.read_csv(results_dir + file))

    return pd.concat(dfs)

def get_dataset(train=True, invariant=False):
    """ Load and convert dataset into inputs and targets """
    model_config = config.get_model_configuration()
    if invariant:
        T = transforms.Compose([
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()]),
            transforms.RandomRotation((0, 360)),
            transforms.ToTensor(),
        ])
    else:
        T = transforms.ToTensor()
    ds_load_function = config.get_global_configuration()['dataset']

    dataset = ds_load_function(os.getcwd(), train=train, download=True, transform=T)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=model_config.get("batch_size"), shuffle=True, num_workers=1)

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
