import numpy as np
import matplotlib.pyplot as plt
import os
import secrets
import string
import json
import scipy.sparse as sp

from skimage.io import imsave
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

from SSA._SSA import photoswitch
from .bin_ssa import bin_ssa
from ..utils import *
from ..psf.psf3d.psf3d import *
from perlin_noise import PerlinNoise

class TimeSeries3D:
    def __init__(self,config):
        self.config = config
        self.particles = config['particles']
        self.dt = config['dt']
        self.texp = config['texp']
        self.T = config['T']
        self.nx = config['nx']
        self.ny = config['ny']
        self.nz = config['nz']
        self.eta = config['eta']
        self.gain = np.load(config['gain'])['arr_0']
        self.offset = np.load(config['offset'])['arr_0']
        self.var = np.load(config['var'])['arr_0']
        self.sigma = config['sigma']
        self.N0 = config['N0']
        self.B0 = config['B0']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.zmin = config['zmin']
        self.pixel_size_lateral = config['pixel_size_lateral']
        self.kvec = np.array([config['k12'],config['k23'],config['k34'],config['k21'],config['k31'],config['k41']]) 
        self.kvec = 1e-3*self.kvec      
        self.zhrange = config['zhrange']
        self.pixel_size_axial = 2*self.zhrange/self.nz
        self.theta = np.zeros((5,self.particles))
        self.theta[0,:] = np.random.uniform(10,self.nx-10,self.particles)
        self.theta[1,:] = np.random.uniform(10,self.ny-10,self.particles)
        self.theta[2,:] = np.random.uniform(-self.zhrange/self.pixel_size_axial,self.zhrange/self.pixel_size_axial,self.particles)
        self.theta[3,:] = self.sigma
        self.theta[4,:] = self.N0
        self.nparams,self.nparticles = self.theta.shape
        self.r = int(self.texp/self.dt)

    def ssa(self):
        k12,k23,k34,k21,k31,k41 = self.kvec
        nt = int(round(self.T/self.dt))
        self.state = np.zeros((self.nparticles,4,nt),dtype=np.bool)
        for n in range(self.nparticles):
            x1, x2, x3, x4, times = photoswitch([self.T,k12,k23,k34,k41,k31,k21])
            t_bins, x1_binned, x2_binned, x3_binned, x4_binned = bin_ssa(times,x1,x2,x3,x4,self.dt,self.T)
            self.state[n,0,:] = x1_binned
            self.state[n,1,:] = x2_binned
            self.state[n,2,:] = x3_binned
            self.state[n,3,:] = x4_binned
         
    def generate(self):
        self.ssa()
        self.get_srate(self.state)
        self.get_brate()
        self.signal_electrons = self.shot_noise(self.srate)
        self.backrd_electrons = self.shot_noise(self.brate)
        self.electrons = self.signal_electrons + self.backrd_electrons     
        
        self.signal_adu = self.gain[np.newaxis,:,:]*self.signal_electrons
        self.signal_adu = self.signal_adu.astype(np.int16) #round
        self.backrd_adu = self.gain[np.newaxis,:,:]*self.backrd_electrons
        self.backrd_adu = self.backrd_adu.astype(np.int16) #round
        self.rnoise_adu = self.read_noise()
        self.rnoise_adu = self.rnoise_adu.astype(np.int16) #round
        
        self.adu = self.signal_adu + self.backrd_adu + self.rnoise_adu
        self.boolean_grid()

    def get_brate(self):
        nt = int(round(self.T/self.texp))
        self.brate = np.zeros((nt,self.nx,self.ny),dtype=np.float32)
        nx,ny = self.nx,self.ny
        noise = PerlinNoise(octaves=10,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        for t in range(nt):
            print(f'Background frame {t}')
            self.brate[t] = self.B0*(bg/bg.max())

    def get_srate(self,state,patch_hw=10,zmin=400.0,ab=6e-7):
        nt = int(round(self.T/self.texp))
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        self.srate = np.zeros((nt,self.nx,self.ny),dtype=np.float32)
        self.xyz_np = np.zeros((nt,self.nparticles,3+1))
        for t in range(nt):
            print(f'Signal frame {t}')
            for n in range(self.nparticles):
                x0,y0,z0,sigma,N0 = self.theta[:,n]
                patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
                x0p = x0-patchx; y0p = y0-patchy
                sigma_x = sx(sigma,z0*self.pixel_size_axial,self.zmin,self.alpha)
                sigma_y = sy(sigma,z0*self.pixel_size_axial,self.zmin,self.beta)
                lam = lamx(X,x0p,sigma_x)*lamy(Y,y0p,sigma_y)
                fon = self.state[n,0,t*self.r:self.r*(t+1)]
                fon = np.sum(fon)/len(fon)
                mu = fon*self.texp*self.eta*N0*lam
                self.srate[t,patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
                self.xyz_np[t,n] = [x0,y0,z0,fon]
        
    def boolean_grid(self,upsample=4):
        nt = int(round(self.T/self.texp))
        grid_shape = (self.nx,self.ny,self.nz)
        boolean_grid_sparse = batch_xyz_to_boolean_grid(self.xyz_np,upsample,
                                                        self.pixel_size_lateral,
                                                        self.pixel_size_axial,
                                                        self.zhrange,
                                                        grid_shape)
        self.spikes = boolean_grid_sparse
        indices = boolean_grid_sparse.coalesce().indices()
        values = boolean_grid_sparse.coalesce().values()
        size = boolean_grid_sparse.size()

        # Create a scipy sparse matrix
        print(values)
        sparse_matrix = sp.csc_matrix((values, indices), shape=size)
        
    def shot_noise(self,rate):
        nt,nx,ny = rate.shape
        electrons = np.zeros_like(rate)
        for n in range(nt):
            electrons[n] = np.random.poisson(lam=rate[n])
        return electrons
                
    def read_noise(self):
        nt = int(round(self.T/self.texp))
        noise_adu = np.zeros((nt,self.nx,self.ny))
        for n in range(nt):
            noise = np.random.normal(self.offset,np.sqrt(self.var),size=(self.nx,self.ny))
            noise_adu[n] += noise
            noise_adu[n] = np.clip(noise_adu[n],0,None)
        return noise_adu
        
    def save(self):
        datapath = self.config['datapath']
        characters = string.ascii_lowercase + string.digits
        unique_id = ''.join(secrets.choice(characters) for i in range(8))
        fname = 'Sim_' + unique_id
        imsave(datapath+fname+'-adu.tif',self.adu,imagej=True)
        imsave(datapath+fname+'-signal_adu.tif',self.signal_adu,imagej=True)
        imsave(datapath+fname+'-backrd_adu.tif',self.backrd_adu,imagej=True)
        imsave(datapath+fname+'-rnoise_adu.tif',self.rnoise_adu,imagej=True)
        with open(datapath+fname+'.json', 'w') as f:
            json.dump(self.config, f)
        np.savez(datapath+fname+'_spikes.npz',spikes=self.spikes)
        np.savez(datapath+fname+'_ssa.npz',state=self.state)
        np.savez(datapath+fname+'_xyz_np.npz',xyz_np=self.xyz_np)

