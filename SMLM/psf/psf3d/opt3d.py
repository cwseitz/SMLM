import numpy as np
import matplotlib.pyplot as plt
from .psf3d import *
from .jac3d import *
from .ill3d import *
from .hess3d import *
from numpy.linalg import inv

class MLEOptimizer3D:
   def __init__(self,theta0,adu,cmos_params,dfcs_params,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.cmos_params = cmos_params
       self.dfcs_params = dfcs_params
       self.theta_gt = theta_gt
   def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,4,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x')
       ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y')
       ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,3])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel(r'$\sigma_x$')
       ax[2].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
       ax[3].plot(thetat[:,4])
       ax[3].set_xlabel('Iteration')
       ax[3].set_ylabel(r'$\sigma_y$')
       ax[3].hlines(y=self.theta_gt[4],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
   def optimize(self,iters=1000,lr=None,plot=False):
       if plot:
           thetat = np.zeros((iters,6))
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0,0])
       loglike = np.zeros((iters,))
       theta = np.zeros_like(self.theta0)
       theta += self.theta0
       for n in range(iters):
           loglike[n] = isologlike3d(theta,self.adu,self.cmos_params,self.dfcs_params)
           jac = jaciso_auto3d(theta,self.adu,self.cmos_params,self.dfcs_params)
           theta = theta - lr*jac
           if plot:
               thetat[n,:] = theta
       if plot:
           self.plot(thetat,iters)
       return theta, loglike
  
class SGLDOptimizer3D:
    def __init__(self,theta0,adu,cmos_params,dfcs_params,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.cmos_params = cmos_params
       self.dfcs_params = dfcs_params
       self.theta_gt = theta_gt
    def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,4,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x')
       ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y')
       ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,2])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel(r'$\sigma_x$')
       ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       ax[3].plot(thetat[:,3])
       ax[3].set_xlabel('Iteration')
       ax[3].set_ylabel(r'$\sigma_y$')
       ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
    def exponential_decay(self, initial_lr, decay_rate, step):
        return initial_lr * np.exp(-decay_rate * step)
    def optimize(self,iters=1000,lr0=None,tburn=0,plot=False):
        ntheta = len(self.theta0)
        thetat = np.zeros((iters,ntheta))
        thetat[0,:] = self.theta0
        if lr0 is None:
            lr0 = np.array([0.0001,0.0001,0.0001,0.0001,0])
        t = np.arange(0,iters,1)
        plt.plot(t,self.exponential_decay(0.01, 0.005, t))
        plt.show()
        for n in range(1,iters):
            jac = jaciso3d(thetat[n-1,:],self.adu,self.cmos_params,self.dfcs_params)
            lr = self.exponential_decay(lr0,0.005,n)
            #lr = lr0
            eps1 = np.random.normal(0,1); std1 = 0.02
            eps2 = np.random.normal(0,1); std2 = 0.02
            eps4 = np.random.normal(0,1); std3 = 0.02
            eps5 = np.random.normal(0,1); std4 = 0.02
            thetat[n,0] = thetat[n-1,0] - lr[0]*jac[0] + std1*eps1
            thetat[n,1] = thetat[n-1,1] - lr[1]*jac[1] + std2*eps2
            thetat[n,2] = thetat[n-1,2] - lr[2]*jac[2] + std3*eps4
            thetat[n,3] = thetat[n-1,3] - lr[3]*jac[3] + std4*eps5
            thetat[n,4] = thetat[n-1,4]
        thetat = thetat[tburn:,:]
        if plot:
            self.plot(thetat,iters-tburn)
       
