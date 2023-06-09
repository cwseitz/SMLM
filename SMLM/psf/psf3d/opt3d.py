import numpy as np
import matplotlib.pyplot as plt
from .psf3d import *
from .jac3d import *
from .ill3d import *

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
           jac = jaciso3d(theta,self.adu,self.cmos_params,self.dfcs_params)
           theta = theta - lr*jac
           if plot:
               thetat[n,:] = theta
       if plot:
           self.plot(thetat,iters)
       return theta, loglike
       
