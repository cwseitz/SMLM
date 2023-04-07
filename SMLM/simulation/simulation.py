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
from SSA.utils import bin_ssa

class Simulation:
    def __init__(self,config,patch_hw=10):
    
        self.config = config
        self.k12 = config['k12']
        self.k23 = config['k23']
        self.k34 = config['k34']
        self.k21 = config['k21']
        self.k31 = config['k31']
        self.k41 = config['k41']
        self.particles = config['particles']
        self.dt = config['dt']
        self.texp = config['texp']
        self.T = config['T']
        self.nx = config['nx']
        self.ny = config['ny']
        self.eta = config['eta']
        self.gain = config['gain']*np.ones((self.nx,self.ny))
        self.offset = config['offset']*np.ones((self.nx,self.ny))
        self.var = config['var']*np.ones((self.nx,self.ny))
        self.sigma = config['sigma']
        self.N0 = config['N0']
        self.B0 = config['B0']
        self.pixel_size = config['pixel_size']
        self.kvec = 1e-3*np.array([self.k12,self.k23,self.k34,self.k21,self.k31,self.k41])
        self.theta = np.zeros((5,self.particles))
        self.theta[0,:] = np.random.uniform(10,self.nx-10,self.particles)
        self.theta[1,:] = np.random.uniform(10,self.ny-10,self.particles)
        self.theta[2,:] = self.sigma
        self.theta[3,:] = self.N0
        self.theta[4,:] = self.B0
        self.nparams,self.nparticles = self.theta.shape
        self.patch_hw = patch_hw
        self.r = int(self.texp/self.dt)
        
    def ssa(self):
        print('Simulating photoswitching with SSA...')
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
            print('\n')
        print('Done.')
        return state  
        
    def simulate(self):
        state = self.ssa()
        nt = int(round(self.T/self.texp))
        movie = np.zeros((nt,self.nx,self.ny),dtype=np.int16)
        x = np.arange(0,2*self.patch_hw); y = np.arange(0,2*self.patch_hw)
        X,Y = np.meshgrid(x,y)
        patch_hw = self.patch_hw
        for t in range(nt):
            print(f'Simulating frame {t}')
            for n in range(self.nparticles):
                x0,y0,sigma,N0,B0 = self.theta[:,n]
                patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
                x0 -= patchx; y0 -= patchy
                alpha = np.sqrt(2)*sigma
                lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
                lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
                lam = lambdx*lambdy
                fon = state[n,0,t*self.r:self.r*(t+1)] #fraction of exposure time the particle was ON
                fon = np.sum(fon)/len(fon)
                s = self.texp*self.eta*(N0*lam*fon)
                electrons = np.random.poisson(lam=s)
                adu = self.gain[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw]*electrons
                adu = adu.astype(np.int16)
                movie[t,patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += adu
            movie[t] += np.random.poisson(lam=self.texp*self.eta*B0,size=(self.nx,self.ny))
            movie[t] = self.read_noise(movie[t])
        return movie, state
                
    def read_noise(self,adu):
        noise = np.random.normal(self.offset,np.sqrt(self.var),size=adu.shape)
        adu += noise.astype(np.int16)
        return adu
        
    def ensemble_average(self,state):
        t_bins = np.arange(0,self.T,self.dt)
        x1avg = np.mean(state[:,0,:],axis=0)
        x2avg = np.mean(state[:,1,:],axis=0)
        x3avg = np.mean(state[:,2,:],axis=0)
        x4avg = np.mean(state[:,3,:],axis=0)
        plt.plot(t_bins,x1avg,label=r'$x_{0}$',color='red')
        plt.plot(t_bins,x2avg,label=r'$x_{1}$',color='blue')
        plt.plot(t_bins,x3avg,label=r'$x_{2}$',color='purple')
        plt.plot(t_bins,x4avg,label=r'$x_{3}$',color='black')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.show()

    def save(self,movie,state):
        datapath = self.config['datapath']
        characters = string.ascii_lowercase + string.digits
        unique_id = ''.join(secrets.choice(characters) for i in range(8))
        spath = 'Sim_' + unique_id
        os.mkdir(datapath+spath)
        imsave(datapath+spath+'/'+spath+'.tif',movie,imagej=True)
        with open(datapath+spath+'/'+'config.json', 'w') as f:
            json.dump(self.config, f)
        np.savez(datapath+spath+'/'+spath+'_ssa.npz',state=state)

