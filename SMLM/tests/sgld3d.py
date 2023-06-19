import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf.psf3d import *

class SGLD3D_Test:
    """Test a single instance of MLE for 3D psf"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
              
    def test(self):
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['z0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        self.thetagt[2] = np.random.normal(0,100)
        iso3d = Iso3D(self.thetagt,self.setup_params)
        theta0 = np.zeros_like(self.thetagt)
        theta0[0] = self.thetagt[0] + np.random.normal(0,2)
        theta0[1] = self.thetagt[1] + np.random.normal(0,2)
        theta0[2] = 0.0
        theta0[3] = self.thetagt[3]
        theta0[4] = self.thetagt[4]
        adu = iso3d.generate(plot=True)
        adu = adu - self.cmos_params[5]
        lr = np.array([0.0001,0.0001,1.0,0,0]) #hyperpar
        opt = SGLDSampler3D(theta0,adu,self.setup_params,lr=lr,theta_gt=self.thetagt)
        samples = opt.sample(iters=1000,plot=True)
        return samples



