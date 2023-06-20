import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import *

class Figure_2:
    """Stochastic gradient langevin dynamics on 3D isolated emitter data"""
    def __init__(self,setup3d_params):
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, ax = plt.subplots(2,3,figsize=(6,3))
        test = SGLD3D_Test(self.setup3d_params)
        samples,mle_est = test.test()
        self.theta_gt = test.thetagt
        self.add_iters(samples,mle_est,ax)
        self.add_hists(samples,mle_est,ax)
        plt.tight_layout()
    def add_iters(self,samples,mle_est,ax):
       iters,nparam = samples.shape
       ax[0,0].plot(samples[:,0],color='pink')
       ax[0,0].set_xlabel('Iteration')
       ax[0,0].set_ylabel('x')
       ax[0,0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='gold',label='gt')
       ax[0,0].hlines(y=mle_est[0],xmin=0,xmax=iters,color='red',label='mle',linestyle='--')
       ax[0,1].plot(samples[:,1],color='darkorchid')
       ax[0,1].set_xlabel('Iteration')
       ax[0,1].set_ylabel('y')
       ax[0,1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='gold')
       ax[0,1].hlines(y=mle_est[1],xmin=0,xmax=iters,color='red',linestyle='--')
       ax[0,2].plot(samples[:,2],color='cornflowerblue')
       ax[0,2].set_xlabel('Iteration')
       ax[0,2].set_ylabel('z')
       ax[0,2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='gold')
       ax[0,2].hlines(y=mle_est[2],xmin=0,xmax=iters,color='red',linestyle='--')
    def add_hists(self,samples,mle_est,ax):
       ax[1,0].hist(samples[:,0],bins=100,color='pink',density=True)
       ax[1,0].set_xticks([])
       ax[1,0].set_ylabel('pmf')
       ax[1,0].set_xlabel('x')
       ax[1,0].vlines(x=self.theta_gt[0],ymin=0,ymax=20,color='gold',label='gt')
       ax[1,0].vlines(x=mle_est[0],ymin=0,ymax=20,color='red',label='mle',linestyle='--')
       ax[1,0].legend()
       ax[1,1].hist(samples[:,1],bins=100,color='darkorchid',density=True)
       ax[1,1].set_ylabel('pmf')
       ax[1,1].set_xlabel('y')
       ax[1,1].vlines(x=self.theta_gt[1],ymin=0,ymax=20,color='gold')
       ax[1,1].vlines(x=mle_est[1],ymin=0,ymax=20,color='red',linestyle='--')
       ax[1,1].set_xticks([])
       ax[1,2].hist(samples[:,2],bins=100,color='cornflowerblue',density=True)
       ax[1,2].set_ylabel('pmf')
       ax[1,2].set_xlabel('z')
       ax[1,2].vlines(x=self.theta_gt[2],ymin=0,ymax=0.2,color='gold')
       ax[1,2].vlines(x=mle_est[2],ymin=0,ymax=0.2,color='red',linestyle='--')
       ax[1,2].set_xticks([])
        

