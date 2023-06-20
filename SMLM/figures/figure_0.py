import numpy as np
import matplotlib.pyplot as plt
from SMLM.utils import *
from SMLM.figures import format_ax

class Figure_0:
    """Quality of Poisson approximation to convolution distribution"""
    def __init__(self,setup2d_params):
        self.setup2d_params = setup2d_params
    def plot(self):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(4.5,2))
        self.add_cmfs(ax1)
        self.add_kdist_var(ax2)
        plt.tight_layout()
    def add_cmfs(self,ax):
        offset = 100
        std = np.sqrt(5)
        rate = 500
        get_kdist(offset,rate,std,ax=ax)
        ax.set_ylabel('Cumulative probability')
        ax.set_xlabel('ADU')
        ax.set_title(r'$\mu_{k}$ = 500')
    def add_kdist_var(self,ax):
        offset = 100
        std = np.sqrt(5)
        rate_space = np.linspace(100,5000,100) #mu
        dist_space = np.zeros_like(rate_space)
        for i,this_rate in enumerate(rate_space):
            dist_space[i] = get_kdist(offset,this_rate,std)
        ax.plot(rate_space,dist_space,color='cornflowerblue')
        ax.set_xlabel(r'$\mu_{k}$')
        ax.set_ylabel('Kolmogorov distance')

        

