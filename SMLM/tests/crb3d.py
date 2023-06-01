import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *
from scipy.stats import multivariate_normal

class CRB3D_Test1:
    """Fixed axial position and variable SNR"""
    def __init__(self):
        self.L = 20
        self.omat = np.ones((self.L,self.L))
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 100.0
        self.gain = self.gain0*self.omat
        self.offset = self.offset0*self.omat
        self.var = self.var0*self.omat
        self.pixel_size = 108.3
        self.sigma = 0.22*640/1.4 
        self.sigma = self.sigma/self.pixel_size
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.B0 = 0
        self.zmin = 400.0
        self.alpha = 6e-7
        self.beta = 6e-7
        self.x0 = 10.0
        self.y0 = 10.0
        self.z0 = 5.0
        self.thetagt = np.array([self.x0,self.y0,self.z0*self.pixel_size,self.sigma,self.N0])
        self.cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        self.dfcs_params = [self.zmin,self.alpha,self.beta]
        
    def plot1(self,nn=5):
        N0space = np.linspace(500,1000,nn)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        rmse = self.rmse_mle_batch(N0space)
        fig, ax = plt.subplots(figsize=(3,4))
        ax.loglog(N0space,rmse[:,0],color='red',marker='x',label='x')
        ax.loglog(N0space,rmse[:,1],color='blue',marker='x',label='y')
        ax.loglog(N0space,rmse[:,2],color='purple',marker='x',label='z')
        ax.set_xlabel('Photons')
        ax.set_ylabel('Localization error (pixels)')
        plt.legend()
        plt.tight_layout()

    def plot2(self):
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        z0space = np.linspace(-5,5,10)
        rmse = self.rmse_mle_batch(z0space)
        fig, ax = plt.subplots(figsize=(3,4))
        ax.loglog(N0space,rmse[:,0],color='red',marker='x',label='x')
        ax.loglog(N0space,rmse[:,1],color='blue',marker='x',label='y')
        ax.loglog(N0space,rmse[:,2],color='purple',marker='x',label='z')
        ax.set_xlabel('Photons')
        ax.set_ylabel('Localization error (pixels)')
        plt.legend()
        plt.tight_layout()

    def rmse_mle3d(self,n0,error_samples=500):
        err = np.zeros((error_samples,5))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[4] = n0
        for n in range(error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.eta,self.texp,self.L,self.gain,self.offset,self.var,self.B0)
            adu = iso3d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,1)
            opt = MLEOptimizer3D(theta0,adu,self.cmos_params,self.dfcs_params)
            theta_est,loglike = opt.optimize(iters=1000)
            err[n,:] = theta_est - self.thetagt
            del iso3d
        return np.sqrt(np.var(err,axis=0))
           

    def rmse_mle_batch(self,N0space):
        errs = np.zeros((len(N0space),5))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_mle3d(n0)
        return errs
 
     
    def crlb(self,N0space,theta0,nn=5):
        crlb_n0 = np.zeros((nn,5))
        for i,n0 in enumerate(N0space):
            theta0[3] = n0
            crlb_n0[i] = crlb3d(theta0,self.cmos_params,self.dfcs_params)
        return crlb_n0

class CRLB3D_Test2:
    """Fixed SNR and variable axial position"""
    def __init__(self):
        self.L = 20
        self.omat = np.ones((self.L,self.L))
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 100.0
        self.gain = self.gain0*self.omat
        self.offset = self.offset0*self.omat
        self.var = self.var0*self.omat
        self.pixel_size = 108.3
        self.sigma = 0.22*640/1.4 
        self.sigma = self.sigma/self.pixel_size
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.B0 = 0
        self.zmin = 400.0
        self.alpha = 6e-7
        self.beta = 6e-7
        self.x0 = 10.0
        self.y0 = 10.0
        self.z0 = 5.0
        self.thetagt = np.array([self.x0,self.y0,self.z0*self.pixel_size,self.sigma,self.N0])
        self.cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        self.dfcs_params = [self.zmin,self.alpha,self.beta]
        
    def plot(self):
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        z0space = np.linspace(-10,10,10)
        #rmse = self.rmse_mle_batch(z0space)
        crlb_z0 = self.crlb(z0space,theta0)
        fig, ax = plt.subplots(figsize=(3,4))
        #ax.plot(self.pixel_size*z0space,self.pixel_size*rmse[:,0],color='red',marker='x',label='x')
        #ax.plot(self.pixel_size*z0space,self.pixel_size*rmse[:,1],color='blue',marker='x',label='y')
        #ax.plot(self.pixel_size*z0space,self.pixel_size*rmse[:,2],color='purple',marker='x',label='z')
        ax.plot(self.pixel_size*z0space,self.pixel_size*crlb_z0[:,0],color='red',linestyle='--')
        ax.plot(self.pixel_size*z0space,self.pixel_size*crlb_z0[:,1],color='blue',linestyle='--')
        ax.plot(self.pixel_size*z0space,self.pixel_size*crlb_z0[:,2],color='purple',linestyle='--')
        ax.set_xlabel('z (nm)')
        ax.set_ylabel('Localization error (nm)')
        plt.legend()
        plt.tight_layout()
        
    def rmse_mle3d(self,z0,error_samples=200):
        err = np.zeros((error_samples,5))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[2] = z0
        for n in range(error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.eta,self.texp,self.L,self.gain,self.offset,self.var,self.B0)
            adu = iso3d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,1)
            opt = MLEOptimizer3D(theta0,adu,self.cmos_params,self.dfcs_params)
            theta_est,loglike = opt.optimize(iters=300)
            err[n,:] = theta_est - self.thetagt
            del iso3d
        return np.sqrt(np.var(err,axis=0))
           

    def rmse_mle_batch(self,z0space):
        errs = np.zeros((len(z0space),5))
        for i,z0 in enumerate(z0space):
            errs[i] = self.rmse_mle3d(z0)
        return errs
 
     
    def crlb(self,z0space,theta0):
        nz = len(z0space)
        crlb_z0 = np.zeros((nz,5))
        for i,z0 in enumerate(z0space):
            theta0[2] = z0
            crlb_z0[i] = crlb3d(theta0,self.cmos_params,self.dfcs_params)
        return crlb_z0
   
class CRLB3D_Test3:
    """Fixed SNR, fixed axial position, variable zmin, variable A/B"""
    def __init__(self):
        self.L = 20
        self.omat = np.ones((self.L,self.L))
        self.gain0 = 2.2
        self.offset0 = 0.0
        self.var0 = 200.0
        self.gain = self.gain0*self.omat
        self.offset = self.offset0*self.omat
        self.var = self.var0*self.omat
        self.pixel_size = 108.3
        self.sigma = 0.22*640/1.4 
        self.sigma = self.sigma/self.pixel_size
        self.texp = 1.0
        self.eta = 0.8
        self.N0 = 1000
        self.B0 = 0
        self.zmin = 400.0
        self.alpha = 6e-7
        self.beta = 6e-7
        self.x0 = 10.0
        self.y0 = 10.0
        self.z0 = 5.0
        self.thetagt = np.array([self.x0,self.y0,self.z0*self.pixel_size,self.sigma,self.N0])
        self.cmos_params = [self.L,self.eta,self.texp,self.gain,self.var]
        self.dfcs_params = [self.zmin,self.alpha,self.beta]
        
    def plot(self):
        zminspace = np.linspace(10,1000,100)
        abspace = np.linspace(1e-7,1e-5,100)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        crlb_map = self.crlb(zminspace,abspace,theta0)
        
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(crlb_map[:,:,0],cmap='jet')
        ax[0].set_title('x')
        ax[0].set_xlabel(r'$z_{min}$')
        ax[0].set_ylabel(r'$\alpha$')
        ax[1].imshow(crlb_map[:,:,1],cmap='jet')
        ax[1].set_title('y')
        ax[1].set_xlabel(r'$z_{min}$')
        ax[1].set_ylabel(r'$\alpha$')
        ax[2].imshow(crlb_map[:,:,2],cmap='jet')
        ax[2].set_title('z')
        ax[2].set_xlabel(r'$z_{min}$')
        ax[2].set_ylabel(r'$\alpha$')
        plt.tight_layout()
        plt.show()
        
    def crlb(self,zminspace,abspace,theta0):
        crlb_map = np.zeros((100,100,5))
        for i,zmin in enumerate(zminspace):
            for j,ab in enumerate(abspace):
                print(f'CRLB Map [{i},{j}]')
                crlb_map[i,j] = crlb3d(theta0,self.cmos_params,self.dfcs_params)
        return crlb_map     
