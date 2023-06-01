import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf
from .psf3d import *

def isologlike3d(theta,adu,eta,texp,gain,var,zmin,alpha,beta):
    lx, ly = adu.shape
    x0,y0,z0,sigma,N0 = self.theta
    x = np.arange(0,self.L); y = np.arange(0,self.L)
    X,Y = np.meshgrid(x,y)
    sigma_x = sx(sigma,z0,zmin,alpha)
    sigma_y = sy(sigma,z0,zmin,beta)
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    stirling = np.nan_to_num(adu*np.log(adu)) - adu
    p = adu*np.log(muprm)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
