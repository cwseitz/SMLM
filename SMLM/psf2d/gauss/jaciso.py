import numpy as np
from .jac1 import *
from .jac2 import *
from .ill import *
from .ill_auto import *


def jaciso(theta,adu,cmos_params):
    lx, ly = adu.shape
    ntheta = len(theta)
    x0,y0,sigma,N0 = theta
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jacobian1(X,Y,x0,y0,sigma,N0,*cmos_params)
    J1 = J1.reshape((ntheta,lx**2))
    J2 = jacobian2(adu,X,Y,x0,y0,sigma,N0,*cmos_params)
    J = J1 @ J2
    return J
    

def jaciso_auto(theta,adu,cmos_params):
    eta,texp,gain,var = cmos_params
    ntheta = len(theta)
    theta = theta.reshape((ntheta,))
    ill = isologlike_auto(adu,eta,texp,gain,var)
    jacobian_ = jacobian(ill)
    jac = jacobian_(theta)
    return jac

 

