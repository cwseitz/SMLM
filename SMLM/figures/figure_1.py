import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import *

class Figure_1:
    """Performance of classical fitting methods on 2D/3D isolated emitter data"""
    def __init__(self,setup2d_params,setup3d_params):
        self.setup2d_params = setup2d_params
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, ax = plt.subplots(2,3,figsize=(7,4))
        #self.add_2d_snr(ax[0,0])
        self.add_3d_snr(ax[0,1])
        #self.add_3d_axi(ax[0,2])
        #self.add_2d_sgld(ax[1,0])
        #self.add_3d_sgld(ax[1,2])
        plt.tight_layout()
    def add_2d_snr(self,ax):
        test = CRB2D_Test1(self.setup2d_params)
        test.plot(ax)
    def add_2d_sgld(self,ax):
        test = SGLD2D_Test(self.setup2d_params)
        samples = test.test()
        test.plot(ax,samples)
    def add_3d_snr(self,ax):
        lr = np.array([0.0001,0.0001,1.0,0,0])
        iters = 200
        error_samples = 500
        test = CRB3D_Test1(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax)
    def add_3d_axi(self,ax):
        lr = np.array([0.0001,0.0001,1.0,0,0])
        iters = 200
        error_samples = 100
        test = CRB3D_Test2(self.setup3d_params,lr,iters=iters,error_samples=error_samples)
        test.plot(ax)
    def add_3d_sgld(self,ax):
        pass
        

