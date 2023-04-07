import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, multivariate_normal

class Distribution:
    def __init__(self):
        pass

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



