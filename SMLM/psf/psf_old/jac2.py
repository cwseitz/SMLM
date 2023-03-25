import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def jacobian2(counts,X,Y,x0,y0,sigma,N0,B0,eta,texp):
    alpha = np.sqrt(2)*sigma
    lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
    lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
    lam = lambdx*lambdy
    I0 = eta*texp*N0
    jac = I0 - (N0*counts)/(N0*lam + B0)
    return jac.flatten()

