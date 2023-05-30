import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf3d import crlb3d
from SMLM.torch.pred import *
from numpy.random import beta
from scipy.stats import multivariate_normal

class CRLB3D:
    def __init__(self):
        self.L = 20
        self.omat = np.ones((self.L,self.L))
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 1.0
        self.gain = self.gain0*self.omat
        self.offset = self.offset0*self.omat
        self.var = self.var0*self.omat
        self.pixel_size = 108.3
        self.sigma = 0.22*640/1.4 
        self.sigma = self.sigma/self.pixel_size
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.B0 = 0
        self.thetagt = np.array([10.0,10.0,0.0,self.sigma,self.N0])

    def plot1(self,nn=5):
        N0space = np.linspace(100,1000,nn)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        #crlb_n0 = self.crlb(N0space,theta0)
        rmse = self.rmse_mle_batch(N0space)
        fig, ax = plt.subplots(figsize=(3,4))
        ax.loglog(N0space,rmse[:,0],color='red',marker='x',label='x')
        ax.loglog(N0space,rmse[:,1],color='blue',marker='x',label='y')
        ax.loglog(N0space,rmse[:,2],color='purple',marker='x',label='z')
        ax.set_xlabel('Photons')
        ax.set_ylabel('Localization error (pixels)')
        plt.legend()
        plt.tight_layout()

    def rmse_mle3d(self,n0,error_samples=500):
        err = np.zeros((error_samples,5))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[3] = n0
        for n in range(error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.eta,self.texp,self.L,self.gain,self.offset,self.var,self.B0)
            cmos_params = [self.eta,self.texp,self.gain,self.var]
            adu = iso3d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,1)
            opt = MLEOptimizer3D(theta0,adu,cmos_params)
            theta_est,loglike = opt.optimize(iters=1000)
            err[n,:] = theta_est - self.thetagt
            del iso3d
        return np.sqrt(np.var(err,axis=0))
           

    def rmse_mle_batch(self,N0space):
        errs = np.zeros((len(N0space),5))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_mle3d(n0)
        return errs
 
     
    def crlb(self,N0space,theta0,nn=5):
        crlb_n0 = np.zeros((nn,5))
        for i,n0 in enumerate(N0space):
            theta0[3] = n0
            crlb_n0[i] = crlb3d(theta0,self.L,self.eta,self.texp,self.gain,self.var)
        return crlb_n0


   
        
