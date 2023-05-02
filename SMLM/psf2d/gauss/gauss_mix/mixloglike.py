import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf
from ..frame import Frame

def mixloglike(theta,adu,eta,texp,gain,var):
    lx, ly = adu.shape
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    mu = np.zeros_like(adu)
    ntheta,nspots = theta.shape
    for n in range(nspots):
        x0,y0,sigma,N0 = theta[:,n]
        alpha = np.sqrt(2)*sigma
        lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
        lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
        lam = lamdx*lamdy
        mu += eta*texp*N0*lam
    stirling = adu*np.log(adu) - adu
    nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
    nll = np.sum(nll)
    return nll


