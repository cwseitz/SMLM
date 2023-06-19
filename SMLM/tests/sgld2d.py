import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf.psf2d import *

class SGLD2D_Test:
    """Test a single instance of SGLD for 2D PSF"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                            setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']] 
    def plot(self,ax,samples):
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color='black')

        ax_histx = ax.twinx()
        ax_histx.hist(samples[:, 0], color='blue', alpha=0.3, bins=50)
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.set_ylim(0, ax_histx.get_ylim()[1] * 1.2)
        ax_histx.tick_params(axis='y', length=0)

        ax_histy = ax.twiny()
        ax_histy.hist(samples[:, 1], color='blue', alpha=0.3, bins=50, orientation='horizontal')
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.set_xlim(0, ax_histy.get_xlim()[1] * 1.2)
        ax_histy.tick_params(axis='x', length=0)

    def test(self):
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        iso2d = Iso2D(self.thetagt,self.setup_params)
        adu = iso2d.generate(plot=False)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,2)
        theta0[1] += np.random.normal(0,2)
        samp = SGLDSampler2D(theta0,adu,self.cmos_params,theta_gt=self.thetagt)
        samples = samp.sample(iters=5000,lr=0.001,tburn=500,plot=False)
        return samples


