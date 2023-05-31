import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.special import erf
from math import floor
from .hess1 import hessian1
from .hess2 import hessian2 
from .jac1 import jacobian1
from .jac2 import jacobian2
from .defocus import *

def crlb3d(theta,L,eta,texp,gain,var,zmin=413.741,ab=5.349139e-7):
    ntheta = len(theta)
    x0,y0,z0,sigma,N0 = theta
    sigma_x, sigma_y = defocus_func(z0,sigma,zmin,ab)
    alpha_x = np.sqrt(2)*sigma_x
    alpha_y = np.sqrt(2)*sigma_y
    x = np.arange(0,L); y = np.arange(0,L)
    X,Y = np.meshgrid(x,y)
    lambdx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
    lam = lambdx*lambdy
    mu = texp*eta*N0*lam + 1e-8
    mu = gain*mu + var
    J = jacobian1(X,Y,x0,y0,z0,sigma,N0,eta,texp,gain,var)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/mu)
    return np.sqrt(np.diagonal(inv(I)))



