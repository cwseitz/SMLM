import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *
from scipy.stats import multivariate_normal

class CRB3D_Test1:
    """Fixed axial position and variable SNR"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['z0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']]) 
        self.dfcs_params = [self.setup_params['zmin'],
                            self.setup_params['alpha'],
                            self.setup_params['beta']]
        self.thetagt[2] = np.random.normal(0,100)
        
    def plot(self,ax,nn=5):
        N0space = np.linspace(100,1000,nn)
        rmse = self.rmse_mle_batch(N0space)
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
            iso3d = Iso3D(theta,self.setup_params)
            adu = iso3d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] = 0
            opt = MLEOptimizer3D(theta0,adu,self.setup_params,theta_gt=theta)
            lr = np.array([0.001,0.001,1.0,0,0])
            theta_est,loglike = opt.optimize(iters=1000,lr=lr,plot=True)
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

class CRB3D_Test2:
    """Fixed SNR and variable axial position"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['z0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']]) 
        self.dfcs_params = [self.setup_params['zmin'],
                            self.setup_params['alpha'],
                            self.setup_params['beta']]
        
    def plot(self,ax):
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        z0space = self.pixel_size*np.linspace(-4,4,10)
        rmse = self.rmse_mle_batch(z0space)
        crlb_z0 = self.crlb(z0space,theta0)
        ax.plot(z0space,self.pixel_size*rmse[:,0],color='red',marker='x',label='x')
        ax.plot(z0space,self.pixel_size*rmse[:,1],color='blue',marker='x',label='y')
        ax.plot(z0space,self.pixel_size*rmse[:,2],color='purple',marker='x',label='z')
        ax.plot(z0space,self.pixel_size*crlb_z0[:,0],color='red',linestyle='--')
        ax.plot(z0space,self.pixel_size*crlb_z0[:,1],color='blue',linestyle='--')
        ax.plot(z0space,crlb_z0[:,2],color='purple',linestyle='--')
        ax.set_xlabel('z (nm)')
        ax.set_ylabel('Localization error (nm)')
        plt.legend()
        plt.tight_layout()
        
    def rmse_mle3d(self,z0,error_samples=100):
        err = np.zeros((error_samples,5))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[2] = z0
        for n in range(error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.setup_params)
            adu = iso3d.generate(plot=False)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,1)
            opt = MLEOptimizer3D(theta0,adu,self.cmos_params,self.dfcs_params,theta)
            theta_est,loglike = opt.optimize(iters=200,plot=True)
            err[n,:] = theta_est - theta
            del iso3d
        plt.hist(err[:,2])
        plt.show()
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
   
class CRB3D_Test3:
    """Fixed SNR, fixed axial position, variable zmin, variable A/B"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['z0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']]) 
        self.dfcs_params = [self.setup_params['zmin'],
                            self.setup_params['alpha'],
                            self.setup_params['beta']]
        
    def plot(self,ax):
        zminspace = np.linspace(10,1000,100)
        abspace = np.linspace(1e-7,1e-5,100)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        crlb_map = self.crlb(zminspace,abspace,theta0)
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
