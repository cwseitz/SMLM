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
    def forward(self,npframe,show=False):
        npframe = npframe.astype(np.float32)
        frame = torch.unsqueeze(torch.from_numpy(npframe),1)
        frame = frame.to(self.device)
        output = self.model(frame)
        xyz_pred = self.pprocessor.forward(output)
        if show:
            fig,ax=plt.subplots(1,3)
            conf_vol = self.pprocessor.get_conf_vol(output)
            mx = np.squeeze(output.cpu().detach().numpy())
            npvol = np.squeeze(conf_vol.cpu().detach().numpy())
            mxvol = np.max(npvol,axis=0)
            mx = np.max(mx,axis=0)
            ax[0].imshow(np.squeeze(npframe),cmap='gray')
            ax[1].imshow(mx,cmap='gray')
            ax[2].imshow(mxvol,cmap='gray')
            plt.show()
        return xyz_pred
        
