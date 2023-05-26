import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm, multivariate_normal
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

class Iso2D:
    def __init__(self,theta,eta,texp,L,gain,offset,var):
        self.theta = theta
        self.gain = gain #ADU/e-
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
        self.L = L

    def generate(self,plot=False):
        rate = self.rate_map()
        electrons = self.shot_noise(rate)           
        adu = self.gain*electrons
        adu = self.read_noise(adu)
        adu = adu.astype(np.int16) #digitize
        if plot:
            self.show(rate,electrons,adu)
        return adu
        
    def rate_map(self):
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
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate) 
        return electrons
                
    def read_noise(self,adu):
        nx,ny = adu.shape
        noise = np.random.normal(self.offset,np.sqrt(self.var),size=(nx,ny))
        adu += noise
        adu = np.clip(adu,0,None)
        return adu
         
    def show(self,rate,electrons,adu):
        fig, ax = plt.subplots(1,3,figsize=(7,2))
        im1 = ax[0].imshow(rate,cmap='gray')
        ax[0].set_xticks([]);ax[0].set_yticks([])
        plt.colorbar(im1, ax=ax[0], label=r'$\mu$')
        im2 = ax[1].imshow(electrons,cmap='gray')
        ax[1].set_xticks([]);ax[1].set_yticks([])
        plt.colorbar(im2, ax=ax[1], label=r'$e^{-}$')
        im4 = ax[2].imshow(adu,cmap='gray')
        ax[2].set_xticks([]);ax[2].set_yticks([])
        plt.colorbar(im4, ax=ax[2], label=r'H (ADU)')
        plt.tight_layout()
        plt.show()

