import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *

class JAC3D_Test:
    """Test analytical jacobian against autograd"""
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
        self.zmin = 400.0
        self.alpha = 6e-7
        self.beta = 6e-7
        self.x0 = 10.0
        self.y0 = 10.0
        self.z0 = 5.0
        self.sigma_x = sx(self.sigma,self.z0,self.zmin,self.alpha)
        self.sigma_y = sy(self.sigma,self.z0,self.zmin,self.beta)
        self.thetagt = np.array([self.x0,self.y0,self.z0*self.pixel_size,self.sigma,self.N0])
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
        cmos_params = [self.L, self.eta,self.texp,self.gain,self.var]
        dfcs_params = [self.zmin,self.alpha,self.beta]
        adu = iso3d.generate(plot=True)
        jac = jaciso3d(self.thetagt,adu,cmos_params,dfcs_params)
        print(jac)
        jac_auto = jaciso_auto3d(self.thetagt,adu,cmos_params,dfcs_params)
        print(jac_auto)
