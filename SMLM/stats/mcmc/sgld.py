import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class SGLD:
    def __init__(self, iters, theta, lr):
        self.iters = iters
        self.theta = theta 
        self.lr = lr
    def iterate(self):
        for i in range(iters):
            mean = np.zeros_like(w)
            std = np.ones_like(w)
            eps1 = np.normal(mean,0.1*std)
            eps2 = np.normal(mean,std)
            if i != 0:
                w.grad.data.zero_()
            loss = L(w)
            loss.backward(np.ones_like(w))
            w_iters.append(w.clone().data.numpy())
            loss_iters.append(loss.data.numpy())
            w.data = w.data - lr*w.grad + lr*eps1 + np.sqrt(lr)*eps2
        return np.asarray(w_iters), np.asarray(loss_iters)     

class NormalLikelihood:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def eval(self,params):
        prod = 1
        for xi,yi in zip(self.x,self.y):
            s = multivariate_normal.pdf(yi,mean=params[1]+params[0]*xi,cov=1)
            prod *= s
        return prod

class NormalPrior:
    def __init__(self,mu,cov):
        self.mu = mu
        self.cov = cov
    def eval(self,m,b):
        f = multivariate_normal(mean=self.mu, cov=self.cov)
        return f.pdf([m,b])
