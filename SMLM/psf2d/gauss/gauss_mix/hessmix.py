import numpy as np
from .jac1mix import *
from .jac2mix import *
from .hess1mix import *
from .hess2mix import *

def hessmix(theta,adu,cmos_params):
    lx, ly = adu.shape
    ntheta, nspots = theta.shape
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jac1mix(X,Y,theta,*cmos_params)
    H1 = hess1mix(adu,X,Y,theta,*cmos_params)
    J2 = jac2mix(adu,X,Y,theta,*cmos_params)
    H2 = hess2mix(X,Y,theta,*cmos_params)
    Ja = J1.reshape((ntheta*nspots,lx**2))
    Hb = H2.reshape((ntheta*nspots,ntheta*nspots,lx**2))
    A = Ja @ H1 @ Ja.T 
    B = np.sum(Hb*J2[np.newaxis, np.newaxis, :],axis=-1)
    H = A + B
    return H
    



