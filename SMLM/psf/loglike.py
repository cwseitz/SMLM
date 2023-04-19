import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def loglike(adu,eta,texp,gain,var):
    lx, ly = adu.shape
    x0,y0,sigma,N0,B0 = theta
    alpha = np.sqrt(2)*sigma
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
    lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
    lam = lamdx*lamdy
    mu = eta*texp*(N0*lam + B0)
    stirling = adu*np.log(adu) - adu
    nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
    nll = np.sum(nll)
    return nll


def mixloglike(adu,eta,texp,gain,var):
    lx, ly = adu.shape
    x0,y0,sigma,N0,B0 = theta
    alpha = np.sqrt(2)*sigma
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
    lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
    lam = lamdx*lamdy
    mu = eta*texp*(N0*lam + B0)
    stirling = adu*np.log(adu) - adu
    nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
    nll = np.sum(nll)
    return nll


