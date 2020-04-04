import torch
from torch.utils.data import Dataset
import numpy as np
from os import path

class NumpyArrayDataset(Dataset):
    def __init__(self, dataset_path, length, min_width=0, transform=None):
        self.length = length
        self.transform = transform
        self.paths = [path.join(dataset_path, f'{i}.npy') for i in range(self.length)]
        
        self.paths = [self.paths[i] for i in range(self.length) if np.load(self.paths[i]).shape[1] >= min_width]
        self.length = len(self.paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = torch.from_numpy(np.load(self.paths[idx]))
        return self.transform(item) if self.transform else item