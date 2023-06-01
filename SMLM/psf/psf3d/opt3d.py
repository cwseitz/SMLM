import numpy as np
import matplotlib.pyplot as plt
from .psf3d import *


class MLEOptimizer3D:
   def __init__(self,theta0,adu,cmos_params):
       self.theta0 = theta0
       self.adu = adu
       self.cmos_params = cmos_params
   def optimize(self,iters=1000,lr=None):
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0,0])
       loglike = np.zeros((iters,))
       theta = np.zeros_like(self.theta0)
       theta += self.theta0
       for n in range(iters):
           loglike[n] = isologlike3d(theta,self.adu,*self.cmos_params)
           jac = jaciso3d(theta,self.adu,self.cmos_params)
           theta = theta - lr*jac
       return theta, loglike
       