import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm, multivariate_normal
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

class GaussianPSF:
    def __init__(self,x0,y0,N0,B0,sigma,eta,texp,L,gain,offset,var,depth=16):
        self.x0 = x0
        self.y0 = y0
        self.L = L
        self.N0 = N0
        self.B0 = B0
        self.alpha = np.sqrt(2)*sigma
        self.gain = gain #ADU/e-
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
    def generate(self,depth=16,plot=False):
        frame = np.zeros((self.L,self.L))
        x = np.arange(0,self.L); y = np.arange(0,self.L)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-self.x0)/self.alpha)-erf((X-0.5-self.x0)/self.alpha))
        lambdy = 0.5*(erf((Y+0.5-self.y0)/self.alpha)-erf((Y-0.5-self.y0)/self.alpha))
        lam = lambdx*lambdy
        s = self.texp*self.eta*(self.N0*lam + self.B0)
        electrons = np.random.poisson(lam=s) #shot noise
        adu = self.gain*electrons
        adu, noise = self.read_noise(adu)
        if plot:
            self.show(s,electrons,noise,adu)
        return adu
    def read_noise(self,adu):
        noise = np.random.normal(self.offset,np.sqrt(self.var),size=adu.shape)
        adu += noise
        return adu,noise
    def show(self,lam,electrons,read_noise,adu):
        fig, ax = plt.subplots(2,2,figsize=(4,3))
        im1 = ax[0,0].imshow(lam,cmap='gray')
        ax[0,0].set_xticks([]);ax[0,0].set_yticks([])
        plt.colorbar(im1, ax=ax[0,0], label=r'$\mu$')
        im2 = ax[0,1].imshow(electrons,cmap='gray')
        ax[0,1].set_xticks([]);ax[0,1].set_yticks([])
        plt.colorbar(im2, ax=ax[0,1], label=r'$e^{-}$')
        im3 = ax[1,0].imshow(read_noise,cmap='gray')
        ax[1,0].set_xticks([]);ax[1,0].set_yticks([])
        plt.colorbar(im3, ax=ax[1,0], label=r'$\xi$ (ADU)')
        im4 = ax[1,1].imshow(adu,cmap='gray')
        ax[1,1].set_xticks([]);ax[1,1].set_yticks([])
        plt.colorbar(im4, ax=ax[1,1], label=r'H (ADU)')
        plt.tight_layout()
        plt.show()




