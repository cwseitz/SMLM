import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial
from SSA._SSA import photoswitch
from SSA.utils import bin_ssa

class Simulation:
    def __init__(self,dt,T,nx,ny,theta,kvec,eta,gain,texp,offset,var,depth=16,patch_hw=10):
        self.dt = dt
        self.T = T
        self.nx = nx
        self.ny = ny
        self.theta = theta
        self.kvec = kvec
        self.gain = gain
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
        self.nparams,self.nparticles = self.theta.shape
        self.patch_hw = patch_hw
        self.r = int(self.texp/self.dt)
    def switch(self):
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
        print('Done.')
        return state    
    def simulate(self,state):
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
        return movie
                
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




