import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial
np.set_printoptions(suppress=True)
np.random.seed(10)


def negloglike_fixed(counts, eta, texp):
    def negloglike_theta(theta,counts=counts):
        lx, ly = counts.shape
        x0,y0,sigma,N0,B0 = theta
        alpha = np.sqrt(2)*sigma
        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
        X = X.ravel(); Y = Y.ravel(); counts = counts.ravel()
        lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
        lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
        I0 = eta*N0*texp
        B = eta*B0*texp
        mu = I0*lamdx*lamdy + B
        counts = counts
        stirling = counts*np.log(counts) - counts
        ll = counts*np.log(mu) - stirling - mu
        ll = np.sum(ll)
        return -1*ll
    return negloglike_theta

def hessian_autograd(theta, counts, eta, texp):
    negloglike_theta = negloglike_fixed(counts, eta, texp)
    hessian_ = hessian(negloglike_theta)
    hess = hessian_(theta)
    return hess




