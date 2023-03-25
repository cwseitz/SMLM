import numpy as np
from scipy.special import erf
from numpy import inf
import matplotlib.pyplot as plt

def hessian1(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp):
    alpha = np.sqrt(2)*sigma
    lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
    lam = lambdx*lambdy
    a = counts*(N0**2)
    b = (N0*lam + B0)**2
    diag = a/b
    diag = diag.flatten()
    hess = np.diag(diag)
    return hess
