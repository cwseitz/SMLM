import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from .psf2d import *
from .ill2d import *
from .jac2d import *
from .hess2d import *
from .lsq2d import *

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
    def show(self,theta0,theta):
        fig,ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta0[0],theta0[1],color='red',label='raw')
        ax.scatter(theta[0],theta[1],color='blue',label='fit')
        ax.legend()
        plt.tight_layout()
        
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        if self.theta_gt is not None:
            ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        
    def grid_search(self,ax,theta,nn=200,delta=3):
        x0,y0,sigma,N0 = theta
        surf = np.zeros((nn,nn))
        x0space = np.linspace(x0-delta,x0+delta,nn)
        y0space = np.linspace(y0-delta,y0+delta,nn)
        X0,Y0 = np.meshgrid(x0space,y0space)
        for i in range(nn):
            for j in range(nn):
                theta_ = np.array([X0[i,j],Y0[i,j],sigma,N0])
                surf[i,j] = isologlike2d(theta_,self.adu,self.cmos_params)
        #ax.contour(X0,Y0,surf,cmap='jet')
        ax.imshow(surf,cmap='jet')
        
    def optimize(self,iters=1000,lr=None,plot=False,grid_search=False):
        thetat = np.zeros((iters,4))
        if lr is None:
            lr = np.array([0.001,0.001,0,0])
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
        if grid_search:
            fig,ax = plt.subplots()
            self.grid_search(ax,theta)
            plt.show()
        if plot:
            self.plot(thetat,iters)
            self.show(self.theta0,theta)
            plt.show()
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
     
class SGLDSampler2D:
    def __init__(self,theta0,adu,cmos_params,theta_gt=None):
        self.theta0 = theta0
        self.adu = adu
        self.cmos_params = cmos_params
        self.theta_gt = theta_gt
    def plot(self,thetat):
        fig,ax = plt.subplots(1,3,figsize=(9,2))
        iters,nparams = thetat.shape
        ax[0].plot(thetat[:,0])
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x (px)')
        if self.theta_gt:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[1].plot(thetat[:,1])
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y (px)')
        if self.theta_gt:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[2].plot(thetat[:,3])
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('N0')
        if self.theta_gt:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        plt.tight_layout()
    def grid_search(self,ax,nn=200,delta=3):
        x0,y0,sigma,N0 = self.theta0
        surf = np.zeros((nn,nn))
        x0space = np.linspace(x0-delta,x0+delta,nn)
        y0space = np.linspace(y0-delta,y0+delta,nn)
        X0,Y0 = np.meshgrid(x0space,y0space)
        for i in range(nn):
            for j in range(nn):
                theta_ = np.array([X0[i,j],Y0[i,j],sigma,N0])
                surf[i,j] = isologlike2d(theta_,self.adu,self.cmos_params)
        ax.contour(X0,Y0,surf,cmap='jet')
    def scatter_samples(self,theta,grid_search=False):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta[:,0],theta[:,1],color='black')
        if grid_search:
            self.grid_search(ax)
        plt.tight_layout()
    def sample(self,iters=1000,lr=None,tburn=0,plot=False):
        ntheta = len(self.theta0)
        thetat = np.zeros((iters,ntheta))
        thetat[0,:] = self.theta0
        if lr is None:
            lr = np.array([0.001,0.001,0,1.0])
        for n in range(1,iters):
            jac = jaciso2d(thetat[n-1,:],self.adu,self.cmos_params)
            eps1 = np.random.normal(0,1)
            eps2 = np.random.normal(0,1)
            eps4 = np.random.normal(0,1)
            thetat[n,0] = thetat[n-1,0] - lr[0]*jac[0] + np.sqrt(lr[0])*eps1
            thetat[n,1] = thetat[n-1,1] - lr[1]*jac[1] + np.sqrt(lr[1])*eps2
            thetat[n,2] = thetat[n-1,2]
            thetat[n,3] = thetat[n-1,3] - lr[3]*jac[3] + np.sqrt(lr[3])*eps4
        thetat = thetat[tburn:,:]
        if plot:
            self.plot(thetat)
            self.scatter_samples(thetat,grid_search=True)
        return thetat
        
class LSQOptimizer2D:
    def __init__(self,adu,setup_params,theta_gt=None):
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
    def optimize(self,spots,plot=False):
        spots, plt_array = fit_psf(self.adu,spots,pltshow=plot,delta=3)
        return spots
