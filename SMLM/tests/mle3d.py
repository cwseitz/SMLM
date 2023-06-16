import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf.psf3d import *

class MLE3D_Test:
    """Test a single instance of MLE for 3D psf"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.dfcs_params = [self.setup_params['zmin'],
                            self.setup_params['alpha'],
                            self.setup_params['beta']]            
    def marginal_likelihood(self,idx,adu,nsamples=100):
        paramgt = self.thetagt[idx]
        bounds = [3,3,0.5,0.5,0.5,300]
        pbound = bounds[idx]
        param_space = np.linspace(paramgt-pbound,paramgt+pbound,nsamples)
        loglike = np.zeros_like(param_space)
        theta = self.thetagt
        for n in range(nsamples):
           theta[idx] = param_space[n]
           loglike[n] = isologlike3d(theta,adu,self.cmos_params,self.dfcs_params)
        fig,ax=plt.subplots()
        ax.plot(param_space,loglike,color='red')
        ax.vlines(paramgt,ymin=loglike.min(),ymax=loglike.max(),color='black')
                   
    def plot_defocus(self):
        fig,ax=plt.subplots()
        zspace = np.linspace(-800,800,100)
        sigma_x = self.sigma + self.alpha*(self.zmin-zspace)**2
        sigma_y = self.sigma + self.beta*(self.zmin+zspace)**2
        ax.plot(zspace,sigma_x,color='red')
        ax.plot(zspace,sigma_y,color='blue')
        ymin,ymax = sigma_x.min(), sigma_x.max()
        ax.vlines(-400,ymin,ymax,color='black',linestyle='--')
        ax.vlines(400,ymin,ymax,color='black',linestyle='--')
        plt.show()
        
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
        #self.plot_defocus()
        lr = np.array([0.0001,0.0001,1.0,0,0]) #hyperpar
        opt = MLEOptimizer3D(theta0,adu,self.setup_params,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(iters=1000,lr=lr,plot=True)

