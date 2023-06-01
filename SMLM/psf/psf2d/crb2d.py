import numpy as np
from numpy.linalg import inv
from .psf2d import *

def crlb2d(theta,L,eta,texp,gain,var):
    ntheta = len(theta)
    x0,y0,sigma,N0 = theta
    alpha = np.sqrt(2)*sigma
    x = np.arange(0,L); y = np.arange(0,L)
    X,Y = np.meshgrid(x,y)
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    J = jac1(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/muprm)
    return np.sqrt(np.diagonal(inv(I)))

