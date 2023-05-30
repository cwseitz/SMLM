import numpy as np
import torch
from os.path import join
from glob import glob
from skimage.io import imread
from torch.utils.data import Dataset

class DetDataset(Dataset):
    def __init__(self,dir):
        self.dir = dir
        files = glob(self.dir+'*.tif')
        inputs = []; targets = []
        for f in files:
            fname = f.split('/')[-1].split('.')[0]
            raw_path = join(self.dir, f)
            tgt_path = join(self.dir, fname + '-mask.npz')
            inputs.append(imread(raw_path))
            targets.append(np.load(tgt_path)['mask'])
        self.inputs = np.concatenate(inputs,axis=0)
        self.targets = np.concatenate(targets,axis=0)
        self.inputs = torch.from_numpy(self.inputs)
        self.inputs = torch.unsqueeze(self.inputs,1)
        self.targets = torch.from_numpy(self.targets)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class RateDataset(Dataset):
    def __init__(self,dir):
        self.dir = dir
        files = glob(self.dir+'*.tif')
        inputs = []; targets = []
        for f in files:
            fname = f.split('/')[-1].split('.')[0]
            raw_path = join(self.dir, f)
            tgt_path = join(self.dir, fname + '-mask.npz')
            inputs.append(imread(raw_path))
            targets.append(np.load(tgt_path)['mask'])
        self.inputs = np.concatenate(inputs,axis=0)
        self.targets = np.concatenate(targets,axis=0)
        self.inputs = torch.from_numpy(self.inputs)
        self.inputs = torch.unsqueeze(self.inputs,1)
        self.targets = torch.from_numpy(self.targets)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
