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

class Generator:
    def __init__(self,config,patch_hw=10):
        self.config = config
        self.particles = config['particles']
        self.dt = config['dt']
        self.texp = config['texp']
        self.T = config['T']
        self.nx = config['nx']
        self.ny = config['ny']
        self.eta = config['eta']
        self.gain = np.load(config['gain'])['arr_0']
        self.offset = np.load(config['offset'])['arr_0']
        self.var = np.load(config['var'])['arr_0']
        self.sigma = config['sigma']
        self.N0 = config['N0']
        self.pixel_size = config['pixel_size']
        self.kvec = np.array([config['k12'],config['k23'],config['k34'],config['k21'],config['k31'],config['k41']]) 
        self.kvec = 1e-3*self.kvec      
        self.theta = np.zeros((4,self.particles))
        self.theta[0,:] = np.random.uniform(10,self.nx-10,self.particles)
        self.theta[1,:] = np.random.uniform(10,self.ny-10,self.particles)
        self.theta[2,:] = self.sigma
        self.theta[3,:] = self.N0
        self.nparams,self.nparticles = self.theta.shape
        self.patch_hw = patch_hw
        self.r = int(self.texp/self.dt)
    def ssa(self):
        k12,k23,k34,k21,k31,k41 = self.kvec
        nt = int(round(self.T/self.dt))
        state = np.zeros((self.nparticles,4,nt),dtype=np.bool)
        for n in range(self.nparticles):
            x1, x2, x3, x4, times = photoswitch([self.T,k12,k23,k34,k41,k31,k21])
            t_bins, x1_binned, x2_binned, x3_binned, x4_binned = bin_ssa(times,x1,x2,x3,x4,self.dt,self.T)
            state[n,0,:] = x1_binned
            state[n,1,:] = x2_binned
            state[n,2,:] = x3_binned
            state[n,3,:] = x4_binned
        return state  
        
    def generate(self):
        state = self.ssa()
        nt = int(round(self.T/self.texp))
        movie = np.zeros((nt,self.nx,self.ny),dtype=np.int16)
        gtmat = []
        x = np.arange(0,2*self.patch_hw); y = np.arange(0,2*self.patch_hw)
        X,Y = np.meshgrid(x,y)
        patch_hw = self.patch_hw
        for t in range(nt):
            print(f'Simulating frame {t}')
            for n in range(self.nparticles):
                x0,y0,sigma,N0 = self.theta[:,n]
                patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
                x0p = x0-patchx; y0p = y0-patchy
                alpha = np.sqrt(2)*sigma
                lambdx = 0.5*(erf((X+0.5-x0p)/alpha)-erf((X-0.5-x0p)/alpha))
                lambdy = 0.5*(erf((Y+0.5-y0p)/alpha)-erf((Y-0.5-y0p)/alpha))
                lam = lambdx*lambdy
                fon = state[n,0,t*self.r:self.r*(t+1)] #fraction of exposure time the particle was ON
                fon = np.sum(fon)/len(fon)
                if fon > 0:
                    gtmat.append([t,fon,x0,y0])
                s = self.texp*self.eta*(N0*lam*fon)
                electrons = np.random.poisson(lam=s)
                adu = self.gain[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw]*electrons
                adu = adu.astype(np.int16)
                movie[t,patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += adu
            movie[t] = self.read_noise(movie[t])
        gtmat = np.array(gtmat)
        return movie, state, gtmat
                
    def read_noise(self,adu):
        noise = np.random.normal(self.offset,np.sqrt(self.var),size=adu.shape)
        adu += noise.astype(np.int16)
        return adu
        
    def save(self,movie,state,gtmat):
        datapath = self.config['datapath']
        characters = string.ascii_lowercase + string.digits
        unique_id = ''.join(secrets.choice(characters) for i in range(8))
        spath = 'Sim_' + unique_id
        os.mkdir(datapath+spath)
        imsave(datapath+spath+'/'+spath+'.tif',movie,imagej=True)
        with open(datapath+spath+'/'+'config.json', 'w') as f:
            json.dump(self.config, f)
        np.savez(datapath+spath+'/'+spath+'_ssa.npz',state=state)
        np.savez(datapath+spath+'/'+spath+'_gtmat.npz',gtmat=gtmat)

