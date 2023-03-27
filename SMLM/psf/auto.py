import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial
np.set_printoptions(suppress=True)
np.random.seed(10)


def negloglike_fixed(adu,eta,texp,gain,var):
    def negloglike_theta(theta,adu=adu,gain=gain,var=var):
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
    return negloglike_theta

def hessian_autograd(theta,adu,eta,texp,gain,var):
    negloglike_theta = negloglike_fixed(adu,eta,texp,gain,var)
    hessian_ = hessian(negloglike_theta)
    hess = hessian_(theta)
    return hess




