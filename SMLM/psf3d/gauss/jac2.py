import numpy as np
from scipy.special import erf
from .defocus import *

def jacobian2(adu,X,Y,x0,y0,z0,sigma,N0,eta,texp,gain,var):
    pixel_size = 108.3
    zmin = 413.741/pixel_size
    a = 5.349139e-7*pixel_size**2
    b = 6.016703e-7*pixel_size**2
    sigma_x = sigma + a*(z0+zmin)**2
    sigma_y = sigma + b*(z0-zmin)**2  
    alpha_x = np.sqrt(2)*sigma_x
    alpha_y = np.sqrt(2)*sigma_y
    lambdx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
    lam = lambdx*lambdy
    mu = gain*eta*texp*N0*lam + var
    jac = 1 - adu/mu
    return jac.flatten()

