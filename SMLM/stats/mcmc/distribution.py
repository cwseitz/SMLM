import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, multivariate_normal

class Distribution:
    def __init__(self):
        pass

class LinearModel(Distribution):
    """Linear model with Gaussian errors"""
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
    def eval(self,params):
        prod = 1
        for xi,yi in zip(self.x,self.y):
            s = multivariate_normal.pdf(yi,mean=params[1]+params[0]*xi,cov=1)
            prod *= s
        return prod

class NormalPrior(Distribution):
    def __init__(self,mu,cov):
        super().__init__()
        self.mu = mu
        self.cov = cov
    def eval(self,m,b):
        f = multivariate_normal(mean=self.mu, cov=self.cov)
        return f.pdf([m,b])

class PoissonNormal(Distribution):
    def __init__(self,mu_norm,mu_psn,sigma_norm):
        self.p = poisson(mu_psn)
        self.g = norm(loc=0, scale=sigma_norm)
        self.mu_norm=mu_norm
        self.mu_psn=mu_psn
        self.sigma_norm=sigma_norm

    def eval(self,x):
        fp = self.p.pmf(x)
        n = x.shape[0]
        mat = np.zeros((n,n))
        for i, fp_i in enumerate(fp):
            mat[i] = (x[i] + self.g.pdf(x))*fp_i
        fpg = np.sum(mat,axis=0)
        return fpg

    def test(self):
        x = np.linspace(-100,100,1000)
        y = self.eval(x)
        plt.plot(x,y,color='black')
        plt.show()

class Normal(Distribution):
    def __init__(self,mu,cov):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.f = multivariate_normal(mean=self.mu,cov=self.cov)
    def eval(self,x):
        vals = self.f.pdf(x)
        return vals
    def sample(self,nsamples):
        return self.f.rvs(size=nsamples)
    def test(self):
        x = np.linspace(-5,5,100)
        y = self.eval(x)
        s = self.sample(1000)
        plt.hist(s,color='blue',alpha=0.5,bins=10,density=True)
        plt.plot(x,y,color='black')
        plt.show()

class Poisson(Distribution):
    def __init__(self,mu):
        super().__init__()
        self.mu = mu
        self.f = poisson(mu)
    def eval(self,x):
        vals = self.f.pmf(x)
        return vals
    def sample(self,nsamples):
        return self.f.rvs(size=nsamples)
    def test(self):
        x = np.arange(0,100,1)
        y = self.eval(x)
        s = self.sample(1000)
        plt.hist(s,color='blue',alpha=0.5,bins=10,density=True)
        plt.plot(x,y,color='black')
        plt.show()

