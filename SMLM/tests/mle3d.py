import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf.psf3d import *

class MLE3D_Test:
    """Test a single instance of MLE for 3D psf"""
    def __init__(self):
        self.L = 20
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 1e-8
        mat = np.ones((self.L,self.L))
        self.gain = self.gain0*mat
        self.offset = self.offset0*mat
        self.var = self.var0*mat
        self.pixel_size = 108.3
        self.sigma = 0.93
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.B0 = 0
        self.zmin = 400.0
        self.alpha = 6e-7
        self.beta = 6e-7
        self.x0 = 10.0
        self.y0 = 10.0
        self.z0 = 400.0
        self.sigma_x = sx(self.sigma,self.z0,self.zmin,self.alpha)
        self.sigma_y = sy(self.sigma,self.z0,self.zmin,self.beta)
        self.thetagt = np.array([self.x0,self.y0,self.sigma,self.sigma_x,self.sigma_y,self.N0])
        self.cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        self.dfcs_params = [self.zmin,self.alpha,self.beta]
        
    def plot_defocus(self):
        fig,ax=plt.subplots()
        zspace = np.linspace(-2*self.zmin,2*self.zmin,100)
        sigma_x = self.sigma + self.alpha*(self.zmin-zspace)**2
        sigma_y = self.sigma + self.beta*(self.zmin+zspace)**2
        ax.plot(zspace,sigma_x,color='red')
        ax.plot(zspace,sigma_y,color='blue')
        ymin,ymax = sigma_x.min(), sigma_x.max()
        ax.vlines(-self.zmin,ymin,ymax,color='black',linestyle='--')
        ax.vlines(self.zmin,ymin,ymax,color='black',linestyle='--')
        plt.show()
        
    def test(self):
        iso3d = Iso3D(self.thetagt,
                      self.eta,
                      self.texp,
                      self.L,
                      self.gain,
                      self.offset,
                      self.var,
                      self.B0)
        theta0 = np.zeros_like(self.thetagt)
        theta0[0] = self.thetagt[0] + np.random.normal(0,1)
        theta0[1] = self.thetagt[1] + np.random.normal(0,1)
        theta0[2] = self.thetagt[2]
        theta0[3] = self.thetagt[2]
        theta0[4] = self.thetagt[2]
        theta0[5] = self.thetagt[5]
        adu = iso3d.generate(plot=True)
        #self.plot_defocus()
        lr = np.array([0.0001,0.0001,0,0.0001,0.0001,0]) #hyperpar
        opt = MLEOptimizer3D(theta0,adu,self.cmos_params,self.dfcs_params,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(iters=100,lr=lr,plot=True)

