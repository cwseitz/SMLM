import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.special import erf
from math import floor
from .hess1 import hessian1
from .hess2 import hessian2 
from .jac1 import jacobian1
from .jac2 import jacobian2

def crlb2d(theta,L,eta,texp,gain,var):
    ntheta = len(theta)
    x0,y0,sigma,N0 = theta
    alpha = np.sqrt(2)*sigma
    x = np.arange(0,L); y = np.arange(0,L)
    X,Y = np.meshgrid(x,y)
    lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
    lam = lambdx*lambdy
    mu = texp*eta*N0*lam + 1e-8
    mu = gain*mu + var
    J = jacobian1(X,Y,x0,y0,sigma,N0,eta,texp,gain,var)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/mu)
    return np.sqrt(np.diagonal(inv(I)))



