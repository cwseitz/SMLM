import autograd.numpy as np
from autograd import grad, jacobian, hessian

def hessmix_auto(theta,adu,cmos_params):
    eta,texp,gain,var = cmos_params
    ntheta, nspots = theta.shape
    theta = theta.T.reshape((ntheta*nspots,))
    ll = mixloglike_auto(adu,eta,texp,gain,var,nspots)
    hessian_ = hessian(ll)
    hess = hessian_(theta)
    return hess


    
