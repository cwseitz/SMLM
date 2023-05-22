import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf

def isologlike(theta,adu,eta,texp,gain,var):
    lx, ly = adu.shape
    x0,y0,sigma,N0 = theta
    alpha = np.sqrt(2)*sigma
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
    lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
    lam = lamdx*lamdy
    mu = eta*texp*N0*lam
    muprm = gain*mu + var
    stirling = np.nan_to_num(adu*np.log(adu)) - adu
    p = adu*np.log(muprm)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
