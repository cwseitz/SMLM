import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

def isologlike_auto3d(adu,eta,texp,gain,var,zmin,alpha,beta):
    def isologlike(theta,adu=adu,gain=gain,var=var):
        lx, ly = adu.shape
        x0,y0,z0,sigma,N0 = theta
        sigma_x = sigma + alpha*(z0+zmin)**2
        sigma_y = sigma + beta*(z0-zmin)**2
        x = np.arange(0,lx); y = np.arange(0,ly)
        X,Y = np.meshgrid(x,y)
        alpha_x, alpha_y = np.sqrt(2)*sigma_x, np.sqrt(2)*sigma_y
        lamx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
        lamy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
        lam = lamx*lamy
        i0 = N0*eta*gain*texp
        muprm = i0*lam + var
        stirling = np.nan_to_num(adu*np.log(adu)) - adu
        p = adu*np.log(muprm)
        p = np.nan_to_num(p)
        nll = stirling + muprm - p
        nll = np.sum(nll)
        return nll
    return isologlike
