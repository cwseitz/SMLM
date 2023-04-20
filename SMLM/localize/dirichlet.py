import numpy as np
from SMLM.psf import *
from SMLM.localize import *
from numpy.random import beta
from scipy.stats import multivariate_normal

class DeconDP:
    def __init__(self,adu,eta,texp,gain,var,alpha=1):
        self.adu = adu
        self.alpha = alpha
        self.cmos_params = (eta,texp,gain,var)
    def dirichlet_process(self,alpha):
        beta_sample = beta(1, alpha)
        weights = [beta_sample]
        while sum(weights) < 1:
            beta_sample = beta(1, alpha)
            weights.append(beta_sample)
        return np.array(weights[:-1])

    def prior(self,prior_params,show=False):
        xvec,yvec,sigma,n0r,b0 = prior_params
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
        
    def run_mcmc(self,prior_params,niter,show=False): 
        data = self.adu
        theta_old = self.prior(prior_params,show=show) #sample from prior       
        like_old = mixloglike(theta_old,data,*self.cmos_params)
        acc = []
        for n in range(niter):
            theta_new = self.prior(prior_params,show=show)
            theta_new, like, accept = self.accrej(theta_old,theta_new,like_old,data,show=show)
            acc.append(accept)

        
    def summarize(self,like_new,like_old,ratio,accept):
        print(f'MCMC iteration:')
        print(f'Like new {like_new}')
        print(f'Like old {like_old}')
        print(f'Acceptance ratio {ratio}')
        print(f'Accepted {accept}\n')


    def accrej(self,theta_old,theta_new,like_old,data,beta=1.0,show=False):
        accept = True
        like_new = mixloglike(theta_new,data,*self.cmos_params)
        a = np.exp(beta*(like_old-like_new))        
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new
            like = like_new
        else:
            accept = False
            theta = theta_old
            like = like_old
            
        self.summarize(like_new,like_old,a,accept)
        return theta, like, accept
        
        



