from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..dataset import SMLMDataset
from .base import *
import numpy as np

class SMLMDataLoader(BaseDataLoader):
    def __init__(self,data_dir,batch_size,shuffle=True,validation_split=0.0,num_workers=1):
        transform = transforms.Compose([
           transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
           transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = SMLMDataset(self.data_dir,transform=transform, target_transform=target_transform)
        super().__init__(self.dataset,batch_size,shuffle,validation_split,num_workers)
  

