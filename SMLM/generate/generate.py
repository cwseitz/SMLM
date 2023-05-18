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

class Generator2D:
    def __init__(self,config):
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
        rate, gtmat = self.rate_map(state)
        electrons = self.shot_noise(rate)           
        adu = self.gain*electrons
        adu = self.read_noise(adu)
        adu = adu.astype(np.int16) #digitize
        return adu, state, gtmat

    def rate_map(self,state,patch_hw=10):
        nt = int(round(self.T/self.texp))
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        rate_map = np.zeros((nt,self.nx,self.ny),dtype=np.float32)
        gtmat = []
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
                fon = state[n,0,t*self.r:self.r*(t+1)]
                fon = np.sum(fon)/len(fon)
                mu = fon*self.texp*self.eta*N0*lam
                rate_map[t,patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
                if fon > 0:
                    gtmat.append([t,fon,x0,y0])     
        gtmat = np.array(gtmat)
        return rate_map, gtmat
        
    def segment(self,gtmat,upsample=1):
        nt = int(round(self.T/self.texp))
        npix = int(upsample*self.nx)
        mask = np.zeros((nt,2,npix,npix),dtype=np.bool)
        for n in range(nt):
           rows = gtmat[gtmat[:,0] == n]
           xpos = rows[:,2]
           ypos = rows[:,3]
           rr = [[0,self.nx],[0,self.nx]]
           vals, xedges, yedges = np.histogram2d(xpos,ypos,bins=npix,range=rr,density=False)
           vals[vals > 1] = 1
           mask[n,0,:,:] = vals
           mask[n,1,:,:] = np.abs(vals-1)
        return mask
   
    def shot_noise(self,rate):
        nt,nx,ny = rate.shape
        electrons = np.zeros_like(rate)
        for n in range(nt):
            electrons[n] = np.random.poisson(lam=rate[n]) 
        return electrons
                
    def read_noise(self,adu):
        nt,nx,ny = adu.shape
        for n in range(nt):
            noise = np.random.normal(self.offset,np.sqrt(self.var),size=(nx,ny))
            adu[n] += noise
            adu[n] = np.clip(adu[n],0,None)
        return adu
        
    def save(self,movie,state,gtmat,mask):
        datapath = self.config['datapath']
        characters = string.ascii_lowercase + string.digits
        unique_id = ''.join(secrets.choice(characters) for i in range(8))
        spath = 'Sim_' + unique_id
        os.mkdir(datapath+spath)
        imsave(datapath+spath+'/'+spath+'.tif',movie,imagej=True)
        imsave(datapath+spath+'/'+spath+'-mask.tif',mask,imagej=True)
        with open(datapath+spath+'/'+'config.json', 'w') as f:
            json.dump(self.config, f)
        np.savez(datapath+spath+'/'+spath+'-mask.npz',mask=mask)
        np.savez(datapath+spath+'/'+spath+'_ssa.npz',state=state)
        np.savez(datapath+spath+'/'+spath+'_gtmat.npz',gtmat=gtmat)

