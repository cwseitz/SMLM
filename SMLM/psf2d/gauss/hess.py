import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from math import floor
from .hess1 import hessian1
from .hess2 import hessian2 
from .jac1 import jacobian1
from .jac2 import jacobian2

def get_errors(theta,adu,eta,texp,gain,var,plot=False):
    ntheta, nspots = theta.shape
    Hbatch = hessian_batch(theta,adu,eta,texp,gain,var)
    Hinv = np.zeros_like(Hbatch)
    errs = np.zeros((nspots,ntheta))
    for n in range(nspots):
        x0,y0,sigma,N0 = theta[:,n]
        x0r = np.floor(x0).astype(np.int)
        y0r = np.floor(y0).astype(np.int)
        sigmar = np.floor(sigma).astype(np.int)
        dx0,dy0 = x0-x0r,y0-y0r
        patch = adu[y0r-5:y0r+5,x0r-5:x0r+5]
        Hinv[n] = np.linalg.inv(Hbatch[n])
        errs[n,:] = np.sqrt(np.diag(Hinv[n]))
        if plot:
            plt.imshow(patch,cmap='gray')
            plt.scatter(5+dx0,5+dy0,marker='x',color='red')
            plt.title(f'{errs[n,:]}')
            plt.show()
    return errs
   
def hessiso(theta,adu,eta,texp,gain,var):
    lx, ly = adu.shape
    ntheta, nspots = theta.shape
    x0,y0,sigma,N0 = theta
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jacobian1(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    H1 = hessian1(adu,X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    J2 = jacobian2(adu,X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    H2 = hessian2(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    Ja = J1.reshape((ntheta,lx**2))
    Hb = H2.reshape((ntheta,ntheta,lx**2))
    A = Ja @ H1 @ Ja.T 
    B = np.sum(Hb*J2[np.newaxis, np.newaxis, :],axis=-1)
    H = A + B
    return H

def hessian_batch(theta,adu,eta,texp,gain,var):
    lx, ly = adu.shape
    ntheta, nspots = theta.shape
    Hbatch = np.zeros((nspots,ntheta,ntheta))
    for n in range(ns):
        x0,y0,sigma,N0 = theta[:,n]
        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
        J1 = jacobian1(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
        H1 = hessian1(adu,X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
        J2 = jacobian2(adu,X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
        H2 = hessian2(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
        Ja = J1.reshape((ntheta,lx**2))
        Hb = H2.reshape((ntheta,ntheta,lx**2))
        A = Ja @ H1 @ Ja.T 
        B = np.sum(Hb*J2[np.newaxis, np.newaxis, :],axis=-1)
        H = A + B
        Hbatch[n,:,:] = H
    return Hbatch



