from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, multivariate_normal
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import special
from scipy.stats import poisson,norm
from scipy.special import j_roots
from scipy.special import beta as beta_fun
from .estimator import *

class NormalMLE(Estimator):
    def __init__(self):
        super().__init__()

    def loglikelihood(self,theta,samples):
        mu, sigma = theta
        f = norm(loc=mu,scale=sigma)
        vals = f.pdf(samples)
        return -np.sum(np.log(vals))

    def optimize(self,samples,method='L-BFGS-B'):
        options = {'disp':True,'maxiter':1000}
        theta0 = (0,1)
        opt = minimize(self.loglikelihood,theta0,args=(samples,),method=method,options=options)
        self.opt_theta = opt.x
        return tuple(self.opt_theta)

class PoissonMLE(Estimator):
    def __init__(self):
        super().__init__()

    def loglikelihood(self,theta,samples):
        mu = theta
        f = poisson(mu)
        vals = f.pmf(samples)
        return -np.sum(np.log(vals))

    def optimize(self,samples,method='L-BFGS-B'):
        options = {'disp':True,'maxiter':1000}
        theta0 = (0,1)
        opt = minimize(self.loglikelihood,theta0,args=(samples,),method=method,options=options)
        self.opt_theta = opt.x
        return tuple(self.opt_theta)

class BurstMLE(Estimator):
    def __init__(self):
        super().__init__()

    def loglikelihood(self,theta,samples):
        kon, koff, ksyn = theta
        at.shape = (len(at), 1)
        np.repeat(at, 50, axis = 1)

        def fun(at, m):
            if(max(m) < 1e6):
                return(poisson.pmf(at,m))
            else:
                return(norm.pdf(at,loc=m,scale=sqrt(m)))

        x,w = j_roots(50,alpha = bet - 1, beta = alpha - 1)
        gs = np.sum(w*fun(at, m = lam*(1+x)/2), axis=1)
        prob = 1/beta_fun(alpha, bet)*2**(-alpha-bet+1)*gs
        return(prob)
        return -np.sum(np.log(vals))

    def optimize(self,samples,method='L-BFGS-B'):
        options = {'disp':True,'maxiter':1000}
        x0 = MomentInference(vals)
        if np.isnan(x0).any() or any(x0 < 0):
            x0 = np.array([10,10,10])
        bnds = ((1e-3,1e3),(1e-3,1e3), (1, 1e4))
        vals_ = np.copy(vals) # Otherwise the structure is violated.
        try:
            ll = minimize(LogLikelihood, x0, args = (vals_), method=metod, bounds=bnds)
        except:
            return np.array([np.nan,np.nan,np.nan])
        #se = ll.hess_inv.todense().diagonal()
        estim = ll.x
        return estim
        opt = minimize(self.loglikelihood,theta0,args=(samples,),method=method,options=options)
        self.opt_theta = opt.x
        return tuple(self.opt_theta)
