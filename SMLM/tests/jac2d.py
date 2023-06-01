import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf.psf2d import *

class JAC2D_Test:
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
        jac = jaciso2d(self.thetagt,adu,cmos_params)
        print(jac)
        jac_auto = jaciso_auto2d(self.thetagt,adu,cmos_params)
        print(jac_auto)
