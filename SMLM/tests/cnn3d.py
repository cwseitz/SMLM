import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from SMLM.generators import Mix3D
from SMLM.torch.utils import prepare_device
from SMLM.torch.models import LocalizationCNN
from SMLM.torch.pred import PostProcessor3D
from skimage.feature import peak_local_max

class CNN3D_Test:
    def __init__(self,setup_config,train_config,pred_config,modelpath,modelname):
        self.modelpath = modelpath
        self.modelname = modelname
        self.train_config = train_config
        self.pred_config = pred_config
        self.model,self.device = self.load_model() 
        pixel_size_axial =  2*setup_config['zhrange']/setup_config['nz']
        self.pprocessor = PostProcessor3D(setup_config,pred_config,device=self.device)
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = self.train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        model = model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
    def forward(self,stack,show=False):
        stack = stack.astype(np.float32)
        stack = torch.unsqueeze(torch.from_numpy(stack),1)
        stack = stack.to(self.device)
        output = self.model(stack)
        if show:
            im1 = torch.squeeze(stack).cpu().detach().numpy()
            im2 = torch.squeeze(output).cpu().detach().numpy()
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(im1)
            ax[1].imshow(im2)
            plt.show()
        #xyz = self.pprocessor.forward(output)
        output = torch.squeeze(output).cpu().detach().numpy()
        coord = peak_local_max(output,threshold_abs=100,min_distance=5)
        return coord
        
class Mix3D_Test:
    def __init__(self,setup_config):
        self.setup_config = setup_config
        self.generator = Mix3D(setup_config)
    def forward(self,nsamples=10,show=False):
        for n in range(nsamples):
            sample, target = self.generator.generate()
            target = target.cpu().detach().numpy()
            target = np.max(target,axis=0)
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(sample[0])
            ax[1].imshow(target)
            plt.show()
