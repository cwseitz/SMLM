import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from glob import glob

class SMLMDataset3D(Dataset):
    def __init__(self,config):
        path = config['data_loader']['path']
        self.config = config
        self.tif_files = glob(path + '*.tif')
        self.npz_files = [f.replace('.tif','.npz') for f in self.tif_files]
    def __len__(self):
        return len(self.tif_files)
    def __getitem__(self, idx):
        adu = imread(self.tif_files[idx])
        x = np.load(self.npz_files[idx])
        spikes = x['spikes']
        theta = x['theta']
        return adu, spikes


