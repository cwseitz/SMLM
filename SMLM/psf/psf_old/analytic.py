import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from .hess1 import hessian1
from .hess2 import hessian2 
from .jac1 import jacobian1
from .jac2 import jacobian2

def hessian_analytical(theta,counts,eta,texp):
    lx, ly = counts.shape
    x0,y0,sigma,N0,B0 = theta
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jacobian1(X,Y,x0,y0,sigma,N0,B0)
    H1 = hessian1(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp)
    J2 = jacobian2(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp)
    H2 = hessian2(X,Y,x0,y0,sigma,N0,B0)
    Ja = J1.reshape((5,lx**2))
    Hb = H2.reshape((5,5,lx**2))
    A = Ja @ H1 @ Ja.T 
    B = np.sum(Hb*J2[np.newaxis, np.newaxis, :],axis=-1)
    H = A + B
    return H

def hessian_analytical_diag(theta,counts,eta,texp):
    lx, ly = counts.shape
    x0,y0,sigma,N0,B0 = theta
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jacobian1(X,Y,x0,y0,sigma,N0,B0)
    H1 = hessian1(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp)
    J2 = jacobian2(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp)
    H2 = hessian2(X,Y,x0,y0,sigma,N0,B0)
    return H1, J1, H2, J2


def negloglike(theta,counts,eta,texp):
    lx, ly = counts.shape
    x0,y0,sigma,N0,B0 = theta
    alpha = np.sqrt(2)*sigma
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    lamdx = 0.5*(erf((X+0.5-x0)*alpha) - erf((X-0.5-x0)*alpha))
    lamdy = 0.5*(erf((Y+0.5-y0)*alpha) - erf((Y-0.5-y0)*alpha))
    I0 = eta*N0*texp
    B = eta*B0*texp
    mu = I0*lamdx*lamdy + B
    counts = counts
    stirling = counts*np.log(counts) - counts
    ll = counts*np.log(mu) - stirling - mu
    ll = np.sum(ll)
    return -1*ll



