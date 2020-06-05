import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


def numpy_data():
    root_dir = '/Users/sroelofs/stanford/cs230/covid-by-county/data/'
    train_x = np.loadtxt(root_dir + 'train/x', dtype = float)
    train_y = np.loadtxt(root_dir + 'train/y', dtype = float)[:, np.newaxis]
    val_x = np.loadtxt(root_dir + 'val/x', dtype = float)
    val_y = np.loadtxt(root_dir + 'val/y', dtype = float)[:, np.newaxis]
    test_x = np.loadtxt(root_dir + 'test/x', dtype = float)
    test_y = np.loadtxt(root_dir + 'test/y', dtype = float)[:, np.newaxis]
    return train_x, train_y, val_x, val_y, test_x, test_y

class CovidData(Dataset):
    def __init__(self, root_dir):
        self.x = torch.Tensor(np.loadtxt(root_dir + "x", dtype = float))
        self.y = torch.Tensor(np.loadtxt(root_dir + "y", dtype = float)[:, np.newaxis])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    

def dataset_loaders(batch_size = 64):
    root_dir = '/Users/sroelofs/stanford/cs230/covid-by-county/data/'
    train_dataset = CovidData(root_dir + 'train/')
    train_load = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_dataset = CovidData(root_dir + 'val/')
    val_load = DataLoader(dataset = val_dataset, batch_size = 10)# len(val_dataset))
    test_dataset = CovidData(root_dir + 'test/')
    test_load = DataLoader(dataset = test_dataset, batch_size = 10)#len(test_dataset))
    return train_load, val_load, test_load
