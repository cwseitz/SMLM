import numpy as np
from SMLM.psf import *
from SMLM.localize import *
from numpy.random import beta
from scipy.stats import multivariate_normal

class DeconDP:
    def __init__(self,adu,eta,texp,gain,var,alpha=1):
        self.adu = adu
        self.eta = eta
        self.texp = texp
        self.gain = gain
        self.var = var
        self.alpha = alpha

    def dirichlet_process(self,alpha):
        beta_sample = beta(1, alpha)
        weights = [beta_sample]
        while sum(weights) < 1:
            beta_sample = beta(1, alpha)
            weights.append(beta_sample)
        return np.array(weights[:-1])

    def prior(self,xvec,yvec,sigma,n0r,b0,show=False):
        nbumps = len(xvec)
        theta = []
        for n in range(nbumps):
            weights = self.dirichlet_process(1)
            mu = np.array([xvec[n],yvec[n]])
            cov = 0.1*np.eye(2)
            nspots = len(weights)
            norm = multivariate_normal(mu,cov)
            n0 = np.random.poisson(lam=n0r) #prior on signal photons
            for m in range(nspots):    
                x0,y0 = norm.rvs() #prior on coordinates
                weight = weights[m]
                theta.append([x0,y0,sigma,n0*weight,b0])
        theta = np.array(theta).T
        if show:
            self.show(theta)
        return theta
        
    def show(self,theta):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta[1,:],theta[0,:],color='red',marker='x')
        plt.show()
        
    def run_mcmc(self,xvec,yvec,sigma,n0r,b0,show=False): 
        theta = self.prior(xvec,yvec,sigma,n0r,b0,show=show)
        nll = mixloglike(theta,self.adu,self.eta,self.texp,self.gain,self.var)
        








