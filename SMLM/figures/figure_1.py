import numpy as np
import matplotlib.pyplot as plt
from SMLM.tests import CRB2D_Test1

class Figure_1:
    """Performance of classical fitting methods on 2D/3D isolated emitter data"""
    def __init__(self,setup2d_params,setup3d_params):
        self.setup2d_params = setup2d_params
        self.setup3d_params = setup3d_params
    def plot(self):
        fig, ax = plt.subplots(2,3)
        self.add_2d_snr(ax[0,0])
        self.add_2d_sgld(ax[1,0])
        self.add_3d_snr(ax[0,1])
        self.add_3d_axial(ax[0,2])
        self.add_3d_hmap(ax[1,1])
        self.add_3d_sgld(ax[1,2])
    def add_2d_snr(self,ax):
        test = CRB2D_Test1(self.setup2d_params)
        test.plot(ax)
    def add_2d_sgld(self,ax):
        pass
    def add_3d_snr(self,ax):
        pass
    def add_3d_axial(self,ax):
        pass
    def add_3d_hmap(self,ax):
        pass
    def add_3d_sgld(self,ax):
        pass
        

