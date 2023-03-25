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

class Estimator():
    def __init__(self):
        pass

class CentroidEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        total = X.sum()
        Xind, Yind = np.indices(X.shape)
        x0 = (Xind*X).sum()/total
        y0 = (Yind*X).sum()/total
        return x0,y0

class Gauss2DMoments(Estimator):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        total = X.sum()
        Xind, Yind = np.indices(X.shape)
        x = (Xind*X).sum()/total
        y = (Yind*X).sum()/total
        col = X[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        row = X[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
        height = X.max()
        return height, x, y, width_x, width_y
