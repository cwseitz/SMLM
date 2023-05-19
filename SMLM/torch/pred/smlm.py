import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from SMLM.torch.utils import prepare_device
from SMLM.torch.models import UNetModel

class SynModel:
    def __init__(self,modelpath,modelname):
        self.modelpath = modelpath
        self.modelname = modelname
    def get_mask(self,sfmx):
        nc,nx,ny = sfmx.shape
        mask = np.zeros((nx,ny))
        mask[sfmx[0,:,:] >= 0.5] = 1
        return mask 
    def apply(self,stack):
        nt,nx,ny = stack.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetModel(1,2)
        model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        prob = np.zeros((nt,2,nx,ny))
        mask = np.zeros((nt,2,nx,ny))
        for n in range(nt):
            frame = stack[n]
            print(f'Applying model to frame {n}')
            with torch.no_grad():
                frame = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                frame = frame.to(device=device, dtype=torch.float)
                output = model(frame).cpu()
                prob[n] = F.softmax(output,dim=1)
                mask[n] = self.get_mask(prob[n])
                torch.cuda.empty_cache()
        return mask

        
        
        
        
