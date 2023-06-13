import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

class SMLMDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.input = torch.from_numpy(imread(path).astype(np.float32))
        self.input = torch.unsqueeze(self.input,1)
    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        #return self.inputs[idx], self.targets[idx]
        return self.input[idx], self.input[idx]


