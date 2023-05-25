import numpy as np
import matplotlib.pyplot as plt
from SMLM.psf2d import jaciso2d, isologlike2d
from SMLM.psf3d import jaciso3d, isologlike3d

class MLEOptimizer2D:
    def __init__(self,theta0,adu,cmos_params):
        self.theta0 = theta0
        self.adu = adu
        self.cmos_params = cmos_params
    def plot(self,theta0,theta):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter([theta0[0]],[theta0[1]],marker='x',color='red')
        ax.scatter([theta[0]],[theta[1]],marker='x',color='blue')
        plt.show()
    def optimize(self,iters=1000,eta=None):
        if eta is None:
            eta = np.array([0.001,0.001,0,0.01])
        iter0 = iters/7
        loglike = np.zeros((iters,))
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        for n in range(iters):
            loglike[n] = isologlike2d(theta,self.adu,*self.cmos_params)
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta[0] -= eta[0]*jac[0]*np.exp(-n/iter0)
            theta[1] -= eta[1]*jac[1]*np.exp(-n/iter0)
            theta[2] -= eta[2]*jac[2]*np.exp(-n/iter0)
            theta[3] -= eta[3]*jac[3]*np.exp(-n/iter0)
        self.plot(self.theta0,theta)
        return theta, loglike
 
class MLEOptimizer3D:
   def __init__(self,theta0,adu,cmos_params):
       self.theta0 = theta0
       self.adu = adu
       self.cmos_params = cmos_params
   def optimize(self,iters=1000,lr=None):
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0,0])
       loglike = np.zeros((iters,))
       theta = np.zeros_like(self.theta0)
       theta += self.theta0
       for n in range(iters):
           loglike[n] = isologlike3d(theta,self.adu,*self.cmos_params)
           jac = jaciso3d(theta,self.adu,self.cmos_params)
           theta = theta - lr*jac
       return theta, loglike
       
class SGLDOptimizer2D:
    def __init__(self,theta0,adu,cmos_params):
        self.theta0 = theta0
        self.adu = adu
        self.cmos_params = cmos_params
    def plot(self,theta0,theta):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter([theta0[0]],[theta0[1]],marker='x',color='red')
        ax.scatter([theta[0]],[theta[1]],marker='x',color='blue')
        plt.show()
    def optimize(self,iters=1000,lr=0.001):
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
        #self.plot(self.theta0,theta)
        return theta
