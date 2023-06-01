import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *

class MLE3D_Test:
    """Test a single instance of MLE for 3D psf"""
    def __init__(self):
        self.L = 20
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 100.0
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
        self.z0 = 5.0
        self.thetagt = np.array([self.x0,self.y0,self.z0*self.pixel_size,self.sigma,self.N0])
        self.cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        self.dfcs_params = [self.zmin,self.alpha,self.beta]
        
    def test(self):
        iso3d = Iso3D(self.thetagt,
                      self.eta,
                      self.texp,
                      self.L,
                      self.gain,
                      self.offset,
                      self.var,
                      self.B0,
                      self.zmin,
                      self.alpha,
                      self.beta)
        theta0 = np.zeros_like(self.thetagt)
        theta0[0] = self.thetagt[0] + np.random.normal(0,1)
        theta0[1] = self.thetagt[1] + np.random.normal(0,1)
        theta0[2] = self.thetagt[2] + np.random.normal(0,1)
        theta0[3] = self.thetagt[3]
        theta0[4] = self.thetagt[4]
        adu = iso3d.generate(plot=True)
        lr = np.array([0.001,0.001,0.001,0,0]) #hyperpar
        opt = MLEOptimizer3D(theta0,adu,self.cmos_params,self.dfcs_params)
        theta, loglike = opt.optimize(iters=200,lr=lr)
        print(self.thetagt)
        print(theta)
