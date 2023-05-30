from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..dataset import DetDataset
from .base import *
import numpy as np

class DetDataLoader(BaseDataLoader):
    def __init__(self,data_dir,batch_size,shuffle=True,validation_split=0.0,num_workers=1):
        self.data_dir = data_dir
        self.dataset = DetDataset(self.data_dir)
        super().__init__(self.dataset,batch_size,shuffle,validation_split,num_workers)
  
class RateDataLoader(BaseDataLoader):
    def __init__(self,data_dir,batch_size,shuffle=True,validation_split=0.0,num_workers=1):
        self.data_dir = data_dir
        self.dataset = RateDataset(self.data_dir)
        super().__init__(self.dataset,batch_size,shuffle,validation_split,num_workers)
