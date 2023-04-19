import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf
from .frame import Frame

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


def mixloglike(theta,adu,eta,texp,gain,var):
    lx, ly = adu.shape
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    mu = np.zeros_like(adu)
    ntheta,nspots = theta.shape
    for n in range(nspots):
        x0,y0,sigma,N0,B0 = theta[:,n]
        alpha = np.sqrt(2)*sigma
        lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
        lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
        lam = lamdx*lamdy
        mu += eta*texp*(N0*lam + B0)
    stirling = adu*np.log(adu) - adu
    nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
    nll = np.sum(nll)
    return nll

# Define the likelihood function as a PyTorch function
def mixloglike_torch(theta, adu, eta, texp, gain, var):
    lx, ly = adu.shape
    X, Y = torch.meshgrid(torch.arange(0,lx), torch.arange(0,ly))
    mu = torch.zeros_like(adu)
    ntheta, nspots = theta.shape
    for n in range(nspots):
        x0, y0, sigma, N0, B0 = theta[:,n]
        alpha = torch.sqrt(torch.tensor(2))*sigma
        lamdx = 0.5*(torch.erf((X+0.5-x0)/alpha) - torch.erf((X-0.5-x0)/alpha))
        lamdy = 0.5*(torch.erf((Y+0.5-y0)/alpha) - torch.erf((Y-0.5-y0)/alpha))
        lam = lamdx*lamdy
        mu += eta*texp*(N0*lam + B0)
    stirling = adu*torch.log(adu) - adu
    nll = stirling + gain*mu + var - adu*torch.log(gain*mu + var)
    nll = torch.sum(nll)
    return -nll
