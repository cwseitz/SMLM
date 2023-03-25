import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Sampler:
    def __init__(self):
        pass
        
class MetropolisHastingsPSF(Sampler):
    def __init__(self,negloglike,mu,cov,eta,texp):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.prop = multivariate_normal(mu,cov)
        self.negloglike = negloglike
        self.eta = eta
        self.texp = texp
        
    def summarize(self,like_new,like_old,theta_new,theta_old,ratio,accept,n):
        print(f'MCMC iteration: {n}')
        print(f'Like new {like_new}')
        print(f'Like old {like_old}')
        print(f'Theta new {theta_new}')
        print(f'Theta old {theta_old}')
        print(f'Acceptance ratio {ratio}')
        print(f'Accepted {accept}\n')

    def thin(self,thetas,skip=5,tburn=500):
        thetas = thetas[:,tburn:]
        thetas = thetas[:,::5]
        return thetas

    def diagnostic(self,thetas,acc,tburn=500):
        acc = np.array(acc)
        acc = acc.astype(np.int)
        f = np.cumsum(acc)
        f = f/np.arange(1,len(f)+1,1)
        #fig, ax = plt.subplots()
        #ax.plot(f,color='black')
        #plt.tight_layout()
        fig, ax = plt.subplots(1,4,figsize=(8,2))
        x0 = thetas[0,tburn:]
        y0 = thetas[1,tburn:]
        sigma = thetas[2,tburn:]
        N0 = thetas[3,tburn:]
        ax[0].plot(x0[::5],color='black')
        ax[1].plot(y0[::5],color='black')
        ax[2].plot(sigma[::5],color='black')
        ax[3].plot(N0[::5],color='black')
        ax[0].set_ylabel(r'$x_{0}$')
        ax[1].set_ylabel(r'$y_{0}$')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        plt.show()
        
        
    def sample(self,theta_old,data,like_old,beta,n):
        accept = True
        dtheta = self.prop.rvs()
        theta_new = theta_old + dtheta
        
        if np.any(theta_new < 0):
            accept = False
            self.summarize(None,like_old,theta_new,theta_old,None,accept,n)
            return theta_old, like_old, accept

        like_new = self.negloglike(theta_new,data,self.eta,self.texp)
        a = np.exp(beta*(like_old-like_new))        
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new
            like = like_new
        else:
            accept = False
            theta = theta_old
            like = like_old
            
        self.summarize(like_new,like_old,theta_new,theta_old,a,accept,n)
        return theta, like, accept
        
    def post_marginals(self,thetas):
        fig, ax = plt.subplots(1,4,figsize=(8,2))
        ax[0].hist(thetas[0,500:],color='black',density=True)
        ax[1].hist(thetas[1,500:],color='black',density=True)
        ax[2].hist(thetas[2,500:],color='black',density=True)
        ax[3].hist(thetas[3,500:],color='black',density=True)
        plt.tight_layout()
        plt.show()
        
        
    def get_betat(self,niter,tau=500):
        iters = np.arange(0,niter,1)
        betat = (1-np.exp(-iters/tau))*0.02
        return betat
        
    def run(self,data,niter,theta0,tburn=500,skip=5):
        theta = theta0
        thetas = np.zeros((len(theta),niter))
        like = self.negloglike(theta0,data,self.eta,self.texp)
        betat = self.get_betat(niter)
        acc = []
        for n in range(niter):
            theta, like, accept = self.sample(theta,data,like,betat[n],n)
            acc.append(accept)
            thetas[:,n] = theta
        #self.diagnostic(thetas,acc)
        thetas = self.thin(thetas,skip=skip,tburn=tburn)
        return thetas
        
           
