import numpy as np
from numpy.linalg import inv
from .psf3d import *

def crlb3d(theta,cmos_params,dfcs_params):
    ntheta = len(theta)
    x0,y0,z0,sigma,N0 = theta
    L,eta,texp,gain,var = cmos_params
    zmin,alpha,beta = dfcs_params
    x = np.arange(0,L); y = np.arange(0,L)
    X,Y = np.meshgrid(x,y)
    sigma_x = sx(sigma,z0,zmin,alpha)
    sigma_y = sy(sigma,z0,zmin,beta)
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    J = jac1(X,Y,theta,cmos_params,dfcs_params)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/muprm)
    return np.sqrt(np.diagonal(inv(I)))



