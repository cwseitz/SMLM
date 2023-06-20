import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

class SMLMDataset3D(Dataset):
    def __init__(self,generator,num_samples):
        self.generator = generator
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        data, target, theta = self.generator.generate()
        return data, target


