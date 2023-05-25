import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf2d import crlb2d
from SMLM.torch.pred import *
from numpy.random import beta
from scipy.stats import multivariate_normal

class Summary2D:
    def __init__(self):
        self.L = 20
        self.omat = np.ones((self.L,self.L))
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 500.0
        self.gain = self.gain0*self.omat
        self.offset = self.offset0*self.omat
        self.var = self.var0*self.omat
        self.pixel_size = 108.3
        self.sigma = 0.22*640/1.4 
        self.sigma = self.sigma/self.pixel_size
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.thetagt = np.array([10.0,10.0,self.sigma,self.N0])
    def sgld(self,tburn=2500):
        fig, ax = plt.subplots(figsize=(4,4))
        dtheta = np.array([0.0,0.0,0.0,0.0])
        theta0 = np.zeros_like(self.thetagt)
        theta0 = self.thetagt + dtheta
        for color,this_var in zip(['dodgerblue','turquoise','pink'],[1,10,100]):
            cmos_params = [self.eta,self.texp,self.gain,this_var]
            iso2d = Iso2D(theta0,self.eta,self.texp,self.L,self.gain,self.offset,this_var)
            adu = iso2d.generate(plot=False)
            opt = SGLDOptimizer2D(theta0,adu,cmos_params)
            theta = opt.optimize(iters=5000,lr=0.002)
            theta = theta[tburn:,:]
            ax.scatter(theta[:,0],theta[:,1],color=color,alpha=0.2,marker='x',label=r'$\sigma_{r}^{2}=$'+f'{this_var}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        plt.tight_layout()
    def crlb(self):
        fig, ax = plt.subplots(figsize=(3,4))
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        for color,this_var in zip(['dodgerblue','turquoise','pink'],[1,10,100]):
            N0space = np.linspace(10,5000,20)
            crlb_n0 = np.zeros((20,4))
            for i,n0 in enumerate(N0space):
                theta0[3] = n0
                crlb_n0[i] = crlb2d(theta0,self.L,self.eta,self.texp,self.gain,this_var)
            ax.loglog(N0space,crlb_n0[:,0],color=color,label=r'$\sigma_{r}^{2}=$'+f'{this_var}')
        ax.set_xlabel('Photons')
        ax.set_ylabel(r'$\sigma_{\mathrm{CRLB}}$ (pixels)')
        plt.legend()
        plt.tight_layout()
              

        


   
        
