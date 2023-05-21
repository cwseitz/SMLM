import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

#def mixloglike_auto(adu,eta,texp,gain,var,nspots):
#    def mixloglike(theta,adu=adu,gain=gain,var=var):
#        lx, ly = adu.shape
#        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
#        mu = np.zeros_like(adu)
#        for n in range(nspots):
#            x0,y0,sigma,N0 = theta[4*n:4*(n+1)]
#            alpha = np.sqrt(2)*sigma
#            lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
#            lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
#            lam = lamdx*lamdy
#            mu += eta*texp*N0*lam
#        stirling = adu*np.log(adu+1e-8) - adu
#        nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
#        nll = np.sum(nll)
#        return nll
#    return mixloglike

def mixloglike_auto(adu,eta,texp,gain,var,nspots,N0,sigma):
    def mixloglike(theta,sigma=sigma,N0=N0,adu=adu,gain=gain,var=var):
        lx, ly = adu.shape
        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
        mu = np.zeros_like(adu)
        for n in range(nspots):
            x0,y0 = theta[2*n:2*(n+1)]
            alpha = np.sqrt(2)*sigma
            lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
            lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
            lam = lamdx*lamdy
            mu += eta*texp*N0*lam
        stirling = adu*np.log(adu+1e-8) - adu
        nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
        nll = np.sum(nll)
        return nll
    return mixloglike

def hessmix_auto(theta,adu,cmos_params):
    eta,texp,gain,var = cmos_params
    ntheta, nspots = theta.shape
    theta = theta.T.reshape((ntheta*nspots,))
    ll = mixloglike_auto(adu,eta,texp,gain,var,nspots)
    hessian_ = hessian(ll)
    hess = hessian_(theta)
    return hess

#temporary function
def jacmix_auto(theta,N0,sigma,adu,cmos_params):
    eta,texp,gain,var = cmos_params
    ntheta, nspots = theta.shape
    theta = theta.reshape((ntheta*nspots,))
    ll = mixloglike_auto(adu,eta,texp,gain,var,nspots,N0,sigma)
    jacobian_ = jacobian(ll)
    jac = jacobian_(theta)
    return jac
    
