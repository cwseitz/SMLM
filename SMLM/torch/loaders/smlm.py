from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..dataset import SMLMDataset3D
from .base import *
import numpy as np

class SMLMDataLoader3D(BaseDataLoader):
    def __init__(self,config,batch_size,shuffle=True,validation_split=0.0,num_workers=1):
        self.dataset = SMLMDataset3D(config)
        super().__init__(self.dataset,batch_size,shuffle,validation_split,num_workers)
  

