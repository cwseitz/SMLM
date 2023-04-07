import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from SSA._SSA import photoswitch
from SSA.utils import *

class Sampler:
    def __init__(self):
        pass
    
class MetropolisHastingsSSA(Sampler):
    def __init__(self,mu,cov):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.prop = multivariate_normal(mu,cov)

    def negloglike(self,theta,X,end_time,dt,Nreps=1000):
        #raw data
        times0, times1 = lifetime4s(X,dt)
        bins = np.arange(0,100,1)
        vals0, vals1 = bin_lifetime(times0,times1,bins)
        #simulation at current params
        vals0sim, vals1sim = self.ssa(theta,end_time,dt,bins,Nreps=Nreps)
        vals0sim = vals0sim + 1e-8 #smooth pmf
        vals1sim = vals1sim + 1e-8 #smooth pmf
        ll = np.sum(vals0*np.log(vals0sim))
        return -1*ll

    def ssa(self,theta,end_time,dt,bins,Nreps=1000):
        print(f'SSA: end_time={end_time}, dt={dt}, Nreps={Nreps}')
        k12,k23,k34,k21,k31,k41 = theta
        nt = int(round(end_time/dt))
        state = np.zeros((Nreps,4,nt),dtype=np.bool)
        for n in range(Nreps):
            x1, x2, x3, x4, times = photoswitch([end_time,k12,k23,k34,k41,k31,k21])
            t_bins, x1_binned, x2_binned, x3_binned, x4_binned =\
            bin_ssa(times,x1,x2,x3,x4,dt,end_time)
            state[n,0,:] = x1_binned
            state[n,1,:] = x2_binned
            state[n,2,:] = x3_binned
            state[n,3,:] = x4_binned
        times0, times1 = lifetime4s(state,dt)
        vals0, vals1 = bin_lifetime(times0,times1,bins,density=True)
        print('Done')
        return vals0, vals1  
          
    def get_betat(self,niter,tau=500):
        iters = np.arange(0,niter,1)
        betat = (1-np.exp(-iters/tau))*0.02
        return betat
              
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
                
    def run(self,X,niter,theta0,dt,end_time,tburn=500,skip=5):
        theta = theta0
        thetas = np.zeros((len(theta),niter))
        like = self.negloglike(theta0,X,end_time,dt)
        print(like)
        betat = self.get_betat(niter)
        #acc = []
        #for n in range(niter):
        #    theta, like, accept = self.sample(theta,data,like,betat[n],n)
        #    acc.append(accept)
        #    thetas[:,n] = theta
        #return thetas
        
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
        
           
