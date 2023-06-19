import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import *

class Figure_2:
    """Stochastic gradient langevin dynamics on 3D isolated emitter data"""
    def __init__(self,setup3d_params):
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, ax = plt.subplots(1,2,figsize=(5,2))
        test = SGLD3D_Test(self.setup3d_params)
        samples = test.test()
        self.add_iters(samples,ax[0])
        self.add_hists(samples,ax[1])
        plt.tight_layout()
    def add_iters(self,samples,ax):
        pass
    def add_hists(self,samples,ax):
        pass

        

