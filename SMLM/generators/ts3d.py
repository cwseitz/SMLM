import numpy as np
import matplotlib.pyplot as plt
import os
import secrets
import string
import json

from skimage.io import imsave
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

from SSA._SSA import photoswitch
from .bin_ssa import bin_ssa
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
        self.pixel_size = config['pixel_size']
        self.kvec = np.array([config['k12'],config['k23'],config['k34'],config['k21'],config['k31'],config['k41']]) 
        self.kvec = 1e-3*self.kvec      
        self.zhrange = 2000.0
        self.theta = np.zeros((5,self.particles))
        self.theta[0,:] = np.random.uniform(10,self.nx-10,self.particles)
        self.theta[1,:] = np.random.uniform(10,self.ny-10,self.particles)
        self.theta[2,:] = np.random.uniform(-self.zhrange,self.zhrange,self.particles)
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
        self.get_spikes()

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
        gtmat = []
        for t in range(nt):
            print(f'Signal frame {t}')
            for n in range(self.nparticles):
                x0,y0,z0,sigma,N0 = self.theta[:,n]
                patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
                x0p = x0-patchx; y0p = y0-patchy
                sigma_x = sx(sigma,z0,self.zmin,self.alpha)
                sigma_y = sy(sigma,z0,self.zmin,self.beta)
                lam = lamx(X,x0p,sigma_x)*lamy(Y,y0p,sigma_y)
                fon = self.state[n,0,t*self.r:self.r*(t+1)]
                fon = np.sum(fon)/len(fon)
                mu = fon*self.texp*self.eta*N0*lam
                self.srate[t,patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
                if fon > 0:
                    gtmat.append([t,fon,x0,y0,z0])     
        self.gtmat = np.array(gtmat)
        
    def get_spikes(self,upsample=4):
        nt = int(round(self.T/self.texp))
        self.spikes = []
        for n in range(nt):
           rows = self.gtmat[self.gtmat[:,0] == n]
           pos = rows[:,2:5]
           rr = [[0,self.nx],[0,self.nx],[-self.zhrange,self.zhrange]]
           bins = [upsample*self.nx,upsample*self.nx,self.nz]
           for point in pos:
               idx = tuple(int(np.floor((p - r[0]) / (r[1] - r[0]) * b)) for p, r, b in zip(point, rr, bins))
               self.spikes.append(idx)
        self.spikes = np.array(self.spikes)
   
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
        np.savez(datapath+fname+'_mask.npz',spikes=self.spikes)
        np.savez(datapath+fname+'_ssa.npz',state=self.state)
        np.savez(datapath+fname+'_gtmat.npz',gtmat=self.gtmat)

