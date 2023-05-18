import numpy as np
from os.path import join
from glob import glob
from skimage.io import imread
from torch.utils.data import Dataset
from SMLM.generate import Generator

class SMLMDataset(Dataset):
	def __init__(self,dir,transform=None,target_transform=None):
		self.dir = dir
		self.transform = transform
		self.target_transform = target_transform
	def __len__(self):
		return len(glob(self.dir+'*.tif'))
	def __getitem__(self, idx):
		files = glob(self.dir+'*.tif')
		f = files[idx]; fname = f.split('.')[0]
		raw_path = join(self.dir, f)
		tgt_path = join(self.dir, fname + '_gtmat.npz')
		stack = imread(raw_path)
		target = np.load(tgt_path)['gtmat']
		if self.transform:
			image = self.transform(stack)
		if self.target_transform:
			targt = self.target_transform(target)
		return image, target

