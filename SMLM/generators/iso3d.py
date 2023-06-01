import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from perlin_noise import PerlinNoise
from ..psf import *

class Iso3D:
    def __init__(self,theta,eta,texp,L,gain,offset,var,B0,zmin,alpha,beta,pixel_size=108.3):
        self.theta = theta
        self.gain = gain #ADU/e-
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
        self.L = L
        self.B0 = B0
        self.pixel_size = pixel_size
        self.zmin = zmin
        self.alpha = alpha
        self.beta = beta
        
    def generate(self,plot=False):
        srate = self.get_srate()
        brate = self.get_brate()
        electrons = self.shot_noise(srate+brate)              
        adu = self.gain*(electrons)
        adu = self.read_noise(adu)
        adu = adu.astype(np.int16) #digitize
        if plot:
            self.show(srate,brate,electrons,adu)
        return adu
        
    def get_srate(self):
        ntheta = self.theta.shape
        x0,y0,z0,sigma,N0 = self.theta
        x = np.arange(0,self.L); y = np.arange(0,self.L)
        X,Y = np.meshgrid(x,y)
        sigma_x = sx(sigma,z0,self.zmin,self.alpha)
        sigma_y = sy(sigma,z0,self.zmin,self.beta)
        lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
        rate = N0*self.texp*self.eta*lam
        return rate

    def get_brate(self):
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
                 
    def show(self,srate,brate,electrons,adu):
    
        fig, ax = plt.subplots(1,4,figsize=(8,1.5))
        
        im1 = ax[0].imshow(srate,cmap='gray')
        ax[0].set_xticks([]);ax[0].set_yticks([])
        plt.colorbar(im1, ax=ax[0], label=r'$\mu_{s}$')
        
        im2 = ax[1].imshow(brate,cmap='gray')
        ax[1].set_xticks([]);ax[1].set_yticks([])
        plt.colorbar(im2, ax=ax[1], label=r'$\mu_{b}$')

        im3 = ax[2].imshow(electrons,cmap='gray')
        ax[2].set_xticks([]);ax[2].set_yticks([])
        plt.colorbar(im1, ax=ax[2], label=r'$e^{-}$')
        
        im4 = ax[3].imshow(adu,cmap='gray')
        ax[3].set_xticks([]);ax[3].set_yticks([])
        plt.colorbar(im4, ax=ax[3], label=r'ADU')
        
        plt.tight_layout()
        plt.show()


