import numpy as np
from os.path import join
from glob import glob
from skimage.io import imread
from torch.utils.data import Dataset

class SMLMDataset(Dataset):
    def __init__(self,dir,transform=None,target_transform=None):
        self.input_dir = dir + 'input/'
        self.target_dir = dir + 'target/'
        self.transform = transform
        self.target_transform = target_transform
        files = glob(self.input_dir+'*.tif')
        inputs = []; targets = []
        for f in files:
            fname = f.split('/')[-1].split('.')[0]
            raw_path = join(self.input_dir, f)
            tgt_path = join(self.target_dir, fname + '-mask.tif')
            inputs.append(imread(raw_path))
            targets.append(imread(tgt_path))
        self.inputs = np.concatenate(inputs,axis=0)
        self.targets = np.concatenate(targets,axis=0)
        if self.transform:
            self.inputs = self.transform(self.inputs).transpose(0,1)
        if self.target_transform:
            self.targets = self.target_transform(self.targets).transpose(0,1)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

