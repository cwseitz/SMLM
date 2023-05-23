import numpy as np
import matplotlib.pyplot as plt
from SMLM.psf2d import jaciso

class Optimizer:
    def __init__(self,theta0,adu,cmos_params):
        self.theta0 = theta0
        self.adu = adu
        self.cmos_params = cmos_params
    def plot(self,theta0,theta):
        fig, ax = plt.subplots()
        ax.imshow(self.adu,cmap='gray')
        ax.scatter([theta0[0]],[theta0[1]],marker='x',color='red')
        ax.scatter([theta[0]],[theta[1]],marker='x',color='blue')
        plt.show()
    def optimize(self,iters=1000,eta=0.001):
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        for n in range(iters):
            jac = jaciso(theta,self.adu,self.cmos_params)
            theta[0] -= eta*jac[0]
            theta[1] -= eta*jac[1]
        self.plot(self.theta0,theta)
        return theta
