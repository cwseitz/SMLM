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
from SMLM.torch.train.metrics import jaccard_coeff
from scipy.spatial.distance import cdist

class CNN3D_Test:
    def __init__(self,setup_config,train_config,pred_config,modelpath,modelname):
        self.modelpath = modelpath
        self.modelname = modelname
        self.setup_config = setup_config
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

    def match_on_xy(self,xyz_pred,xyz_true,threshold=3):
        distances = cdist(xyz_true[:,:2],xyz_pred[:,:2])
        xyz_pred_matched = []
        xyz_true_matched = []
        for i in range(len(xyz_true)):
            for j in range(len(xyz_pred)):
                if distances[i, j] < threshold:
                    xyz_true_matched.append(xyz_true[i])
                    xyz_pred_matched.append(xyz_pred[j])
        xyz_pred_matched = np.array(xyz_pred_matched)
        xyz_true_matched = np.array(xyz_true_matched)
        return xyz_pred_matched, xyz_true_matched


    def show_pred(self,frame,xyz_pred,xyz_true):
        frame = np.squeeze(frame)
        fig,ax = plt.subplots()
        ax.imshow(frame,cmap='gray')
        ax.scatter(xyz_pred[:,1],xyz_pred[:,0],marker='x',color='red')
        ax.scatter(xyz_true[:,1],xyz_true[:,0],marker='x',color='blue')
        plt.show()

    def test(self,num_samples,show=False):
        generator = Mix3D(self.setup_config)
        xyz_true_batch = []; xyz_pred_batch = []
        for n in range(num_samples):
            sample, target, theta = generator.generate()
            xyz_true = theta[:3,:].T
            xyz_pred = self.forward(sample,show=False)
            xyz_pred = xyz_pred.astype(np.float32)
            xyz_pred[:,0] = xyz_pred[:,0]/4.0
            xyz_pred[:,1] = xyz_pred[:,1]/4.0
            if show:
                self.show_pred(sample,xyz_pred,xyz_true)
            xyz_pred_matched, xyz_true_matched = self.match_on_xy(xyz_pred,xyz_true)
            xyz_true_batch.append(xyz_true_matched)
            xyz_pred_batch.append(xyz_pred_matched)
        xyz_true_batch = np.concatenate(xyz_true_batch,axis=0)
        xyz_pred_batch = np.concatenate(xyz_pred_batch,axis=0)
        return xyz_true_batch,xyz_pred_batch

    def get_errors(self,true_batch,pred_batch,num_bins=10):
        min_value = np.min(true_batch)
        max_value = np.max(true_batch)
        bin_indices = np.digitize(true_batch, np.linspace(min_value, max_value, num_bins))
        errors = pred_batch - true_batch
        bin_means = np.zeros(num_bins)
        bin_variances = np.zeros(num_bins)
        for bin_idx in range(num_bins):
            bin_errors = errors[bin_indices == bin_idx]
            bin_means[bin_idx] = np.mean(bin_errors)
            bin_variances[bin_idx] = np.var(bin_errors)
        return bin_means,bin_variances     
