import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from perlin_noise import PerlinNoise

class Iso2D:
    def __init__(self,theta,eta,texp,L,gain,offset,var,B0):
        self.theta = theta
        self.gain = gain #ADU/e-
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
        self.L = L
        self.B0 = B0
        
    def generate(self,plot=False):
        rate1 = self.rate1()
        rate2 = self.rate2()
        electronsfg = self.shot_noise(rate1)      
        electronsbg = self.shot_noise(rate2)           
        adu = self.gain*(electronsbg+electronsfg)
        adu = self.read_noise(adu)
        adu = adu.astype(np.int16) #digitize
        if plot:
            self.show(rate1+rate2,electronsfg,electronsbg,adu)
        return adu
        
    def rate1(self):
        ntheta = self.theta.shape
        x0,y0,sigma,N0 = self.theta
        alpha = np.sqrt(2)*sigma
        x = np.arange(0,self.L); y = np.arange(0,self.L)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
        lam = lambdx*lambdy
        rate = N0*self.texp*self.eta*lam
        return rate

    def rate2(self):
        noise = PerlinNoise(octaves=10,seed=None)
        nx,ny = self.L,self.L
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        bg_rate = self.B0*(bg/bg.max())
        return bg_rate
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate) 
        return electrons
                
    def read_noise(self,adu):
        nx,ny = adu.shape
        noise = np.random.normal(self.offset,np.sqrt(self.var),size=(nx,ny))
        adu += noise
        adu = np.clip(adu,0,None)
        return adu
                 
    def show(self,rate,electronsfg,electronsbg,adu):
    
        fig, ax = plt.subplots(1,4,figsize=(8,1.5))
        
        im1 = ax[0].imshow(rate,cmap='gray')
        ax[0].set_xticks([]);ax[0].set_yticks([])
        plt.colorbar(im1, ax=ax[0], label=r'$\mu$')
        
        im2 = ax[2].imshow(electronsfg,cmap='gray')
        ax[2].set_xticks([]);ax[2].set_yticks([])
        plt.colorbar(im2, ax=ax[2], label=r'$\mathrm{FG} \;e^{-}$')

        im3 = ax[1].imshow(electronsbg,cmap='gray')
        ax[1].set_xticks([]);ax[1].set_yticks([])
        plt.colorbar(im1, ax=ax[1], label=r'$\mathrm{BG} \;e^{-}$')
        
        im4 = ax[3].imshow(adu,cmap='gray')
        ax[3].set_xticks([]);ax[3].set_yticks([])
        plt.colorbar(im4, ax=ax[3], label=r'ADU')
        
        plt.tight_layout()
        plt.show()

