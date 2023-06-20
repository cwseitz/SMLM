import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import *
from SMLM.figures import format_ax

class Figure_1:
    """Performance of classical fitting methods on 3D isolated emitter data"""
    def __init__(self,setup3d_params):
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,2))
        ax1t = ax1.twinx()
        ax2t = ax2.twinx()
        self.add_3d_snr(ax1,ax1t)
        self.add_3d_axi(ax2,ax2t)
        plt.tight_layout()
    def add_3d_snr(self,ax1,ax2):
        lr = np.array([0.0001,0.0001,10.0,0,0])
        iters = 300
        error_samples = 500
        test = CRB3D_Test1(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax1,ax2)
    def add_3d_axi(self,ax1,ax2):
        lr = np.array([0.0001,0.0001,10.0,0,0])
        iters = 300
        error_samples = 500
        test = CRB3D_Test2(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax1,ax2)

        

