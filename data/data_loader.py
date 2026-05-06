import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, size=2000):
        self.X = torch.randn(size, 3, 32, 32)
        self.y = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_loaders():
    train = DummyDataset(2000)
    test = DummyDataset(500)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64)
    return train_loader, test_loader
