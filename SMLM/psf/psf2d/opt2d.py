import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from .psf2d import *
from .ill2d import *
from .jac2d import *
from .hess2d import *

class MLEOptimizer2DGrad:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        plt.show()
    def optimize(self,iters=1000,lr=None,plot=False):
        if plot:
            thetat = np.zeros((iters,4))
        if lr is None:
            lr = np.array([0.001,0.001,0,0.01])
        loglike = np.zeros((iters,))
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        for n in range(iters):
            loglike[n] = isologlike2d(theta,self.adu,self.cmos_params)
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta[0] -= lr[0]*jac[0]
            theta[1] -= lr[1]*jac[1]
            theta[2] -= lr[2]*jac[2]
            theta[3] -= lr[3]*jac[3]
            if plot:
                thetat[n,:] = theta
        if plot:
            self.plot(thetat,iters)
        return theta, loglike
 
class MLEOptimizer2DNewton:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        plt.show()
    def optimize(self,iters=1000,plot=False):
        if plot:
            thetat = np.zeros((iters,4))
        loglike = np.zeros((iters,))
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        for n in range(iters):
            loglike[n] = isologlike2d(theta,self.adu,self.cmos_params)
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            hess = hessiso_auto2d(theta,self.adu,self.cmos_params)
            diag = np.diagonal(hess)
            dd = np.nan_to_num(jac/diag)
            theta = theta - dd
            if plot:
                thetat[n,:] = theta
        if plot:
            self.plot(thetat,iters)
        return theta, loglike
     
class SGLDOptimizer2D:
    def __init__(self,theta0,adu,cmos_params):
        self.theta0 = theta0
        self.adu = adu
        self.cmos_params = cmos_params
    def plot(self,theta0,theta):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter([theta0[0]],[theta0[1]],marker='x',color='red',label='start')
        ax.scatter([theta[0]],[theta[1]],marker='x',color='blue',label='end')
        ax.legend()
        plt.show()
    def scatter_samples(self,theta):
        fig, ax = plt.subplots()
        ax.scatter(theta[:,0],theta[:,1],color='black')
        plt.tight_layout()
        plt.show()
    def optimize(self,iters=1000,lr=0.001,tburn=500,scatter=False):
        ntheta = len(self.theta0)
        theta = np.zeros((iters,ntheta))
        theta[0,:] = self.theta0
        for n in range(1,iters):
            jac = jaciso2d(theta[n-1,:],self.adu,self.cmos_params)
            eps1 = np.random.normal(0,1)
            eps2 = np.random.normal(0,1)
            theta[n,0] = theta[n-1,0] - lr*jac[0] + np.sqrt(lr)*eps1
            theta[n,1] = theta[n-1,1] - lr*jac[1] + np.sqrt(lr)*eps2
            theta[n,2] = theta[n-1,2]
            theta[n,3] = theta[n-1,3]
        theta = theta[tburn:,:]
        if scatter:
            self.scatter_samples(theta)
        theta_est = np.mean(theta,axis=0)
        return theta_est
