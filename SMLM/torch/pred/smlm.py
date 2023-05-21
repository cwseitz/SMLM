import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from SMLM.psf2d import mixloglike, jacmix, jacmix_auto
from SMLM.torch.utils import prepare_device
from SMLM.torch.models import UNetModel

class NeuralEstimator2D:
    def __init__(self,config,datapath,modelpath,modelname,gtpath=None):
        self.config = config
        self.datapath = datapath
        self.gtpath = gtpath
        self.modelpath = modelpath
        self.modelname = modelname
        self.gain = np.load(config['gain'])['arr_0']
        self.offset = np.load(config['offset'])['arr_0']
        self.var = np.load(config['var'])['arr_0']
        self.eta = config['eta']
        self.texp = config['texp']
        self.N0 = config['N0']
        self.sigma = config['sigma']
        self.cmos_params = [self.eta,self.texp,self.gain,self.var]
    def get_mask(self,sfmx):
        nc,nx,ny = sfmx.shape
        mask = np.zeros((nx,ny))
        mask[sfmx[0,:,:] >= 0.5] = 1
        return mask 
    def loadgt(self):
        return np.load(self.gtpath)['gtmat']
    def filter_spots(self,coordinates,xmin,xmax,ymin,ymax):
        x = coordinates[:,0]
        y = coordinates[:,1]
        x_mask = np.logical_and(x >= xmin, x <= xmax)
        y_mask = np.logical_and(y >= ymin, y <= ymax)
        filtered_coordinates = coordinates[np.logical_and(x_mask, y_mask)]
        return filtered_coordinates
    def diagnostic(self,adu,theta,patch_coords,patch_mask):
        print(theta)
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(adu,cmap='gray')
        ax[1].imshow(patch_mask,cmap='gray')
        ax[0].scatter(patch_coords[:,1],patch_coords[:,0],color='red',marker='x')
        plt.show()
    def optimize(self,adu,mask,gtmat=None,patch_hw=10):
        theta0 = np.argwhere(mask > 0)
        nspots = theta0.shape[0]
        theta = theta0
        for n in range(nspots):
            x0,y0 = theta[n,:]
            xmin,xmax = x0-patch_hw,x0+patch_hw
            ymin,ymax = y0-patch_hw,y0+patch_hw
            patch_gt_coords = self.filter_spots(gtmat,xmin,xmax,ymin,ymax)
            patch_coords[:,0] -= xmin
            patch_coords[:,1] -= ymin
            patch_adu = adu[xmin:xmax,ymin:ymax]
            patch_msk = mask[xmin:xmax,ymin:ymax]
            patch_gain = self.gain[xmin:xmax,ymin:ymax]
            patch_var = self.var[xmin:xmax,ymin:ymax]
            self.diagnostic(patch_adu,theta,patch_coords,patch_msk)
            this_theta = np.array([x0,y0,self.N0,self.sigma])
            this_theta = this_theta[:,np.newaxis].T
            cmos_params = [self.eta,self.texp,patch_gain,patch_var]
            jac = jacmix(this_theta.T,patch_adu,cmos_params)
            jac_auto = jacmix_auto(this_theta.T,patch_adu,cmos_params)
            dx0,dy0,ds,dn0 = jac
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetModel(1,2)
        model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device   
    def forward(self,stack):
        nt,nx,ny = stack.shape
        prob = np.zeros((nt,2,nx,ny))
        mask = np.zeros((nt,2,nx,ny))
        if self.gtpath != None:
           gtmat = self.loadgt()
        model,device = self.load_model()
        for n in range(nt):
            frame = stack[n]
            if self.gtpath != None:
                rows = gtmat[gtmat[:,0] == n]
                rows = rows[:,2:]
            else: 
                rows = None
            print(f'Applying model to frame {n}')
            with torch.no_grad():
                frame = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                frame = frame.to(device=device, dtype=torch.float)
                output = model(frame).cpu()
                prob[n] = F.softmax(output,dim=1)
                mask[n] = self.get_mask(prob[n])
                self.optimize(frame.cpu().numpy().squeeze(),mask[n,0],gtmat=rows)
                torch.cuda.empty_cache()
        return mask

        
        
        
        
