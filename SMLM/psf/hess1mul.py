import numpy as np
from scipy.special import erf
from numpy import inf

def hess1mul(adu,X,Y,theta,eta,texp,gain,var):
    alpha = np.sqrt(2)*sigma
    ntheta,nspots = theta.shape
    nlam = np.zeros_like(adu)
    for n in range(nspots):
        x0,y0,sigma,N0,B0 = theta[:,n]
        lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
        lam += N0*lambdx*lambdy
    mu = gain*eta*texp*(nlam + B0) + var
    diag = adu/mu**2
    hess = np.diag(diag.flatten())
    return hess
