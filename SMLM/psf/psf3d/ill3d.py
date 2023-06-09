import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf
from .psf3d import *


def isologlike3d(theta,adu,cmos_params,dfcs_params):
    x0,y0,sigma,sigma_x,sigma_y,N0 = theta
    zmin,alpha,beta = dfcs_params
    L,eta,texp,gain,var = cmos_params
    x = np.arange(0,L); y = np.arange(0,L)
    X,Y = np.meshgrid(x,y)
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    stirling = np.nan_to_num(adu*np.log(adu)) - adu
    p = adu*np.log(muprm)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
