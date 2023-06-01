import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf.psf2d import *

class MLE2D_Test:
    """Test a single instance of maximum likelihood estimation"""
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
        self.B0 = 100
        self.x0 = 10.0
        self.y0 = 10.0
        self.thetagt = np.array([self.x0,self.y0,self.sigma,self.N0])
    def test(self):
        iso2d = Iso2D(self.thetagt,
                      self.eta,
                      self.texp,
                      self.L,
                      self.gain,
                      self.offset,
                      self.var,
                      self.B0)
        cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        adu = iso2d.generate(plot=True)
        lr = np.array([0.001,0.001,0,5.0]) #hyperpar
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,1)
        theta0[1] += np.random.normal(0,1)
        opt = MLEOptimizer2D(theta0,adu,cmos_params)
        theta, loglike = opt.optimize(iters=100,lr=lr,plot=True)
