import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import *

class Figure_1:
    """Performance of classical fitting methods on 3D isolated emitter data"""
    def __init__(self,setup3d_params):
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, ax = plt.subplots(1,2,figsize=(5,2))
        #self.add_3d_snr(ax[0,0])
        self.add_3d_axi(ax[0,1])
        plt.tight_layout()
    def add_3d_snr(self,ax):
        lr = np.array([0.0001,0.0001,1.0,0,0])
        iters = 300
        error_samples = 500
        test = CRB3D_Test1(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax)
    def add_3d_axi(self,ax):
        lr = np.array([0.0001,0.0001,10.0,0,0])
        iters = 300
        error_samples = 500
        test = CRB3D_Test2(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax)

        

