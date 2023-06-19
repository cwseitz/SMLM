import numpy as np
import matplotlib.pyplot as plt
from .psf3d import *
from .jac3d import *
from .ill3d import *
from .hess3d import *
from numpy.linalg import inv

class MLEOptimizer3D:
   def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
       self.dfcs_params = [setup_params['zmin'],setup_params['alpha'],setup_params['beta']]
       self.theta_gt = theta_gt
   def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,3,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x (px)')
       ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y (px)')
       ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,2])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel(r'z (nm)')
       ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
   def optimize(self,iters=1000,lr=None,plot=False):
       if plot:
           thetat = np.zeros((iters,5))
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0,0])
       loglike = np.zeros((iters,))
       theta = np.zeros_like(self.theta0)
       theta += self.theta0
       for n in range(iters):
           loglike[n] = isologlike3d(theta,self.adu,self.cmos_params,self.dfcs_params)
           jac = jaciso3d(theta,self.adu,self.cmos_params,self.dfcs_params)
           theta = theta - lr*jac
           if plot:
               thetat[n,:] = theta
       if plot:
           self.plot(thetat,iters)
       return theta, loglike
  
class SGLDSampler3D:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
       self.dfcs_params = [setup_params['zmin'],setup_params['alpha'],setup_params['beta']]
       self.theta_gt = theta_gt
    def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,3,figsize=(8,2))
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
       ax[2].set_ylabel('z')
       ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
    def sample(self,iters=1000,lr=None,tburn=0,plot=False):
        ntheta = len(self.theta0)
        thetat = np.zeros((iters,ntheta))
        thetat[0,:] = self.theta0
        if lr is None:
            lr = np.array([0.0001,0.0001,2.0,0,0])
        t = np.arange(0,iters,1)
        for n in range(1,iters):
            jac = jaciso3d(thetat[n-1,:],self.adu,self.cmos_params,self.dfcs_params)
            epsx = np.random.normal(0,1)
            epsy = np.random.normal(0,1)
            epsz = np.random.normal(0,1)
            thetat[n,0] = thetat[n-1,0] - lr[0]*jac[0] + np.sqrt(lr[0])*epsx
            thetat[n,1] = thetat[n-1,1] - lr[1]*jac[1] + np.sqrt(lr[1])*epsy
            thetat[n,2] = thetat[n-1,2] - lr[2]*jac[2] + np.sqrt(lr[2])*epsz
            thetat[n,3] = thetat[n-1,3]
            thetat[n,4] = thetat[n-1,4]
        thetat = thetat[tburn:,:]
        if plot:
            self.plot(thetat,iters-tburn)
        return thetat
       
