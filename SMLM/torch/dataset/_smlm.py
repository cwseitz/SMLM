import numpy as np
from os.path import join
from torch.utils.data import Dataset
from SMLM.generate import Generator

class SMLMDataset(Dataset):
	def __init__(self,dir):
		self.raw_dir = dir + 'raw'
		self.tgt_dir = dir + 'tgt'
	def __len__(self):
		return len(os.listdir(self.raw_dir))
	def __getitem__(self, idx):
		files = os.listdir(self.raw_dir)
		f = files[idx]; fname = f.split('.')[0]
		raw_path = os.path.join(self.raw_dir, f)
		tgt_path = os.path.join(self.raw_dir, fname + '.npz')
		stack = imread(img_path).astype(np.float64)
		target = np.load(tgt_path)['arr_0']
		if self.transform:
			image = self.transform(image)
		return image, target
	def generate(self,n):
		os.makedirs(self.raw_dir, exist_ok=True)
		os.makedirs(self.tgt_dir, exist_ok=True)
		for i in range(n):
		    generator = Generator()

