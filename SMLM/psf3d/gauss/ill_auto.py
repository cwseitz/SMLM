import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

def isologlike_auto3d(adu,eta,texp,gain,var):
    def isologlike(theta,adu=adu,gain=gain,var=var):
        lx, ly = adu.shape
        x0,y0,z0,sigma,N0 = theta
        sigma_x = sigma + 5.349139e-7*(z0+413.741)**2
        sigma_y = sigma + 6.016703e-7*(z0-413.741)**2
        alpha_x = np.sqrt(2)*sigma_x
        alpha_y = np.sqrt(2)*sigma_y
        x = np.arange(0,lx); y = np.arange(0,ly)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))
        lam = lambdx*lambdy
        mu = eta*texp*N0*lam
        muprm = gain*mu + var
        stirling = np.nan_to_num(adu*np.log(adu)) - adu
        p = adu*np.log(muprm)
        p = np.nan_to_num(p)
        nll = stirling + muprm - p
        nll = np.sum(nll)
        return nll
    return isologlike
