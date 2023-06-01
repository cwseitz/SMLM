import numpy as np
from .psf2d import *

def isologlike2d(theta,adu,cmos_params):
    L,eta,texp,gain,var = cmos_params
    x0,y0,sigma,N0 = theta
    X,Y = np.meshgrid(np.arange(0,L),np.arange(0,L))
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    stirling = np.nan_to_num(adu*np.log(adu)) - adu
    p = adu*np.log(muprm)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
