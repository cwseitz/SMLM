import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm, multivariate_normal
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

class GaussianPSF:
    def __init__(self,x0,y0,L,I0,B,sigma,gain,rmu,rvar,depth=16):
        self.x0 = x0
        self.y0 = y0
        self.L = L
        self.I0 = I0
        self.B = B
        self.alpha = 1/(np.sqrt(2)*sigma)
        self.gain = gain #ADU/e-
        self.rmu = rmu
        self.rvar = rvar
    def generate(self,depth=16,plot=False):
        frame = np.zeros((self.L,self.L))
        x = np.arange(0,self.L); y = np.arange(0,self.L)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-self.x0)*self.alpha)-erf((X-0.5-self.x0)*self.alpha))
        lambdy = 0.5*(erf((Y+0.5-self.y0)*self.alpha)-erf((Y-0.5-self.y0)*self.alpha))
        lam = lambdx*lambdy
        s = self.I0*lam + self.B
        electrons = np.random.poisson(lam=s) #shot noise
        adu = self.gain*electrons
        adu, noise = self.read_noise(adu)
        max_adu = 2**depth - 1
        adu = adu.astype(np.int16)
        adu[adu > max_adu] = max_adu
        if plot:
            self.show(lam,electrons,noise,adu)
        return adu
    def read_noise(self,adu):
        noise = np.random.normal(self.rmu,np.sqrt(self.rvar),size=adu.shape)
        adu += noise
        return adu,noise
    def show(self,lam,electrons,read_noise,adu):
        fig, ax = plt.subplots(2,2,figsize=(4,3))
        im1 = ax[0,0].imshow(lam,cmap='gray')
        ax[0,0].set_xticks([]);ax[0,0].set_yticks([])
        ax[0,0].set_title(f'Sum={np.sum(lam)}')
        plt.colorbar(im1, ax=ax[0,0], label=r'$\lambda$')
        im2 = ax[0,1].imshow(electrons,cmap='gray')
        ax[0,1].set_xticks([]);ax[0,1].set_yticks([])
        plt.colorbar(im2, ax=ax[0,1], label=r'$e^{-}$')
        im3 = ax[1,0].imshow(read_noise,cmap='gray')
        ax[1,0].set_xticks([]);ax[1,0].set_yticks([])
        plt.colorbar(im3, ax=ax[1,0], label=r'$\xi$ (ADU)')
        im4 = ax[1,1].imshow(adu,cmap='gray')
        ax[1,1].set_xticks([]);ax[1,1].set_yticks([])
        plt.colorbar(im4, ax=ax[1,1], label=r'ADU')
        plt.tight_layout()
        plt.show()


#class GaussianPSF:
#    def __init__(self,x0,y0,L,I0,sigma,gain,rmu,rvar,depth=16):
#        self.x0 = x0
#        self.y0 = y0
#        self.L = L
#        self.I0 = I0
#        self.alpha = 1/(np.sqrt(2)*sigma)
#        self.gain = gain #ADU/e-
#        self.rmu = rmu
#        self.rvar = rvar
#    def generate(self,depth=16,plot=False):
#        frame = np.zeros((self.L,self.L))
#        x = np.arange(0,self.L); y = np.arange(0,self.L)
#        X,Y = np.meshgrid(x,y)
#        lambdx = 0.5*(erf((X+0.5-self.x0)*self.alpha)-erf((X-0.5-self.x0)*self.alpha))
#        lambdy = 0.5*(erf((Y+0.5-self.y0)*self.alpha)-erf((Y-0.5-self.y0)*self.alpha))
#        lam = lambdx*lambdy
#        electrons = np.random.poisson(lam=self.I0*lam) #shot noise
#        adu = self.gain*electrons
#        adu, noise = self.read_noise(adu)
#        max_adu = 2**depth - 1
#        adu = adu.astype(np.int16)
#        adu[adu > max_adu] = max_adu
#        if plot:
#            self.show(lam,electrons,noise,adu)
#        return adu
#    def read_noise(self,adu):
#        noise = np.random.normal(self.rmu,np.sqrt(self.rvar),size=adu.shape)
#        adu += noise
#        return adu,noise
#    def show(self,lam,electrons,read_noise,adu):
#        fig, ax = plt.subplots(2,2,figsize=(4,3))
#        im1 = ax[0,0].imshow(lam,cmap='gray')
#        ax[0,0].set_xticks([]);ax[0,0].set_yticks([])
#        ax[0,0].set_title(f'Sum={np.sum(lam)}')
#        plt.colorbar(im1, ax=ax[0,0], label=r'$\lambda$')
#        im2 = ax[0,1].imshow(electrons,cmap='gray')
#        ax[0,1].set_xticks([]);ax[0,1].set_yticks([])
#        plt.colorbar(im2, ax=ax[0,1], label=r'$e^{-}$')
#        im3 = ax[1,0].imshow(read_noise,cmap='gray')
#        ax[1,0].set_xticks([]);ax[1,0].set_yticks([])
#        plt.colorbar(im3, ax=ax[1,0], label=r'$\xi$ (ADU)')
#        im4 = ax[1,1].imshow(adu,cmap='gray')
#        ax[1,1].set_xticks([]);ax[1,1].set_yticks([])
#        plt.colorbar(im4, ax=ax[1,1], label=r'ADU')
#        plt.tight_layout()
#        plt.show()


