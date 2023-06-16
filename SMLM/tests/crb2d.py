import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf import *
from scipy.stats import multivariate_normal

class CRB2D_Test1:
    """Variable SNR"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])          
    def plot(self,nn=5):
        N0space = np.linspace(100,1000,nn)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        crlb_n0 = self.crlb(N0space,theta0)
        rmse = self.rmse_mle_batch(N0space)
        fig, ax = plt.subplots(figsize=(3,4))
        ax.loglog(N0space,crlb_n0[:,0],color='red')
        ax.loglog(N0space,rmse[:,0],color='red',marker='x')
        ax.set_xlabel('Photons')
        ax.set_ylabel(r'$\sigma_{\mathrm{CRLB}}$ (pixels)')
        plt.legend()
        plt.tight_layout()
       
    def crlb(self,N0space,theta0,nn=5):
        crlb_n0 = np.zeros((nn,4))
        for i,n0 in enumerate(N0space):
            theta0[3] = n0
            crlb_n0[i] = crlb2d(theta0,self.cmos_params)
        return crlb_n0

    def rmse_sgld_batch(self,N0space):
        errs = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_sgld2d(n0)
        return errs
        
    def rmse_mle_batch(self,N0space):
        errs = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_mle2d(n0)
        return errs
        
    def rmse_mle2d(self,n0,nsamples=500):
        err = np.zeros((nsamples,4))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[3] = n0
        for n in range(nsamples):
            print(f'Error sample {n}')
            iso2d = Iso2D(self.thetagt,self.setup_params)
            adu = iso2d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            opt = MLEOptimizer2DGrad(theta0,adu,self.setup_params,theta_gt=self.thetagt)
            theta_est,loglike = opt.optimize(iters=200)
            err[n,:] = theta_est - self.thetagt
            del iso2d

        return np.sqrt(np.var(err,axis=0))

        


   
        
