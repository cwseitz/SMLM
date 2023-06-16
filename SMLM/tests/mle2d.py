import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf.psf2d import *

class MLE2DGrad_Test:
    """Test a single instance of maximum likelihood estimation"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                            setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
    def test(self):
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        iso2d = Iso2D(self.thetagt,self.setup_params)
        adu = iso2d.generate(plot=True)
        lr = np.array([0.001,0.001,0.0,0.0]) #hyperpar
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,2)
        theta0[1] += np.random.normal(0,2)
        opt = MLEOptimizer2DGrad(theta0,adu,self.setup_params,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(iters=100,lr=lr,plot=True)
        

class MLE2DNewton_Test:
    """Test a single instance of maximum likelihood estimation with Newton-Raphson"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                            setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
    def test(self):
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        iso2d = Iso2D(self.thetagt,self.setup_params)
        adu = iso2d.generate(plot=True)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,1)
        theta0[1] += np.random.normal(0,1)
        opt = MLEOptimizer2DNewton(theta0,adu,self.setup_params,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(iters=10,plot=True)
