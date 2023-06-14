import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from SMLM.torch.utils import prepare_device
from SMLM.torch.models import LocalizationCNN

class NeuralEstimator3D:
    def __init__(self,nz,scaling_factor,dilation_flag,modelpath,modelname):
        self.modelpath = modelpath
        self.modelname = modelname
        self.nz = nz
        self.scaling_factor = scaling_factor
        self.dilation_flag = dilation_flag
        self.model,self.device = self.load_model()  
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LocalizationCNN(self.nz,self.scaling_factor,dilation_flag=self.dilation_flag)
        model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
    def forward(self,stack):
        stack = stack.astype(np.float32)
        stack = torch.unsqueeze(torch.from_numpy(stack),1)
        stack = stack.to(self.device)
        output = self.model(stack)

        
        
        
        
