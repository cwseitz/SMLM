import numpy as np
import matplotlib.pyplot as plt
from .psf2d import *
from .ill2d import *
from .jac2d import *

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
    def optimize(self,iters=1000,lr=None,plot=False):
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
            self.plot(self.theta0,theta)
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
