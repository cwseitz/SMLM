import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module, MaxPool3d, ConstantPad3d
from torch.nn.functional import conv3d

def tensor_to_np(x):
    return np.squeeze(x.cpu().detach().numpy())

class PostProcessor3D(Module):
    def __init__(self,setup_config,pred_config,device='cpu'):
        super().__init__()
        self.setup_config = setup_config
        self.pred_config = pred_config
        self.device = device
        self.thresh = pred_config['thresh']
        self.r = pred_config['radius']
        self.pixel_size_lateral = setup_config['pixel_size_lateral']/4
        self.pixel_size_axial = 2*setup_config['zhrange']/setup_config['nz']
        self.zmin = setup_config['zhrange']/self.pixel_size_axial
        self.upsampling_shift = 0
        self.maxpool = MaxPool3d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

    def get_conf_vol(self,pred_vol):
        pred_thresh = torch.where(pred_vol > self.thresh, pred_vol, self.zero)
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)
        conf_vol = torch.squeeze(conf_vol)
        return conf_vol
 
    def forward(self,pred_vol):
        num_dims = len(pred_vol.size())
        if np.not_equal(num_dims, 5):
            if num_dims == 4:
                pred_vol = pred_vol.unsqueeze(0)
            else:
                pred_vol = pred_vol.unsqueeze(0)
                pred_vol = pred_vol.unsqueeze(0)
                
        conf_vol = self.get_conf_vol(pred_vol)
        batch_idx = torch.nonzero(conf_vol)
        zbool, xbool, ybool = batch_idx[:, 0], batch_idx[:, 1], batch_idx[:, 2]
        xbool, ybool, zbool = tensor_to_np(xbool), tensor_to_np(ybool), tensor_to_np(zbool)
        xrec = xbool; yrec = ybool; zrec = zbool
        zrec = zrec - self.setup_config['zhrange']/self.pixel_size_axial
        xyz_rec = np.column_stack((xrec, yrec, zrec)) + 0.5

        return xyz_rec

