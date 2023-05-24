import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm, multivariate_normal
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

class Iso3D:
    def __init__(self,theta,eta,texp,L,gain,offset,var,depth=16):
        self.theta = theta
        self.gain = gain #ADU/e-
        self.offset = offset
        self.var = var
        self.texp = texp
        self.eta = eta
        self.L = L
        self.adu = np.zeros((self.L,self.L))
        self.read_noise = np.random.normal(self.offset,np.sqrt(self.var),size=self.adu.shape)
        self.electrons = np.zeros((self.L,self.L))
        self.mu = np.zeros((self.L,self.L))
    def generate(self,depth=16,plot=False):
        ntheta = self.theta.shape
        x0,y0,z0,sigma,N0 = self.theta
        sigma_x = sigma + 5.349139e-7*(z0+413.741)**2
        sigma_y = sigma + 6.016703e-7*(z0-413.741)**2
        alpha_x = np.sqrt(2)*sigma_x
        alpha_y = np.sqrt(2)*sigma_y
        x = np.arange(0,self.L); y = np.arange(0,self.L)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
        lam = lambdx*lambdy
        mu = self.texp*self.eta*N0*lam
        self.mu += mu
        electrons = np.random.poisson(lam=self.mu) #shot noise
        self.electrons += electrons
        adu = self.gain*electrons
        self.adu += adu
        self.adu += self.read_noise
        if plot:
            self.show(self.mu,self.electrons,self.read_noise,self.adu)
        return self.adu
    def show(self,rate,electrons,read_noise,adu):
        fig, ax = plt.subplots(2,2,figsize=(8,8))
        im1 = ax[0,0].imshow(rate,cmap='gray')
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


