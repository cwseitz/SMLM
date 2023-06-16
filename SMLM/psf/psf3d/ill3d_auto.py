import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

def lamx(X,x0,sigma_x):
    alpha_x = np.sqrt(2)*sigma_x
    return 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
    
def lamy(Y,y0,sigma_y):
    alpha_y = np.sqrt(2)*sigma_y
    return 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))

def sx(sigma,z0,zmin,alpha):
    return sigma + alpha*(z0+zmin)**2
    
def sy(sigma,z0,zmin,beta):
    return sigma + beta*(z0-zmin)**2

def isologlike_auto3d(adu,eta,texp,gain,var):
    def isologlike(theta,adu=adu,gain=gain,var=var):
        nx,ny = adu.shape
        x0,y0,z0,sigma,N0 = theta
        sigma_x = sx(sigma,z0,zmin,alpha)
        sigma_y = sy(sigma,z0,zmin,beta)
        x = np.arange(0,nx); y = np.arange(0,ny)
        X,Y = np.meshgrid(x,y)
        lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
        i0 = N0*eta*gain*texp
        muprm = i0*lam + var
        stirling = np.nan_to_num(adu*np.log(adu)) - adu
        p = adu*np.log(muprm)
        p = np.nan_to_num(p)
        nll = stirling + muprm - p
        nll = np.sum(nll)
        return nll
    return isologlike
