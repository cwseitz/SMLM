import numpy as np
from scipy.special import erf
from numpy import inf

def hessian1(adu,X,Y,x0,y0,sigma,N0,B0,eta,texp,gain,var):
    alpha = np.sqrt(2)*sigma
    lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
    lam = lambdx*lambdy
    mu = gain*eta*texp*(N0*lam + B0) + var
    diag = adu/mu**2
    hess = np.diag(diag.flatten())
    return hess
