import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from SMLM.torch.utils import prepare_device
from SMLM.torch.models import LocalizationCNN
from .post3d import Postprocess3D

class NeuralEstimator3D:
    def __init__(self,setup_config,train_config,pred_config,modelpath,modelname):
        self.modelpath = modelpath
        self.modelname = modelname
        self.train_config = train_config
        self.pred_config = pred_config
        self.model,self.device = self.load_model() 
        pixel_size_axial =  2*setup_config['zhrange']/setup_config['nz']
        self.pprocessor = Postprocess3D(setup_config,pred_config,device=self.device)
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = self.train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
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
        xyz_rec, conf_rec = self.pprocessor.forward(output)
        return xyz_rec, conf_rec
        
        
        
        
