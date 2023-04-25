import numpy as np
from scipy.special import erf

def jac2mul(adu,X,Y,theta,eta,texp,gain,var):
    alpha = np.sqrt(2)*sigma
    ntheta,nspots = theta.shape
    nlam = np.zeros_like(adu)
    for n in range(nspots):
		lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
		lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
		nlam += N0*lambdx*lambdy
    mu = gain*eta*texp*(nlam + B0) + var
    jac = 1 - adu/mu
    return jac.flatten()


