import numpy as np
from scipy.special import erf
from numpy import inf

def hessian1(adu,X,Y,x0,y0,z0,sigma,N0,eta,texp,gain,var):
    sigma_x = sigma + 5.349139e-7*(z0+413.741)**2
    sigma_y = sigma + 6.016703e-7*(z0-413.741)**2
    alpha_x = np.sqrt(2)*sigma_x
    alpha_y = np.sqrt(2)*sigma_y
    lambdx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
    lam = lambdx*lambdy
    mu = gain*eta*texp*N0*lam + var
    diag = adu/mu**2
    hess = np.diag(diag.flatten())
    return hess
