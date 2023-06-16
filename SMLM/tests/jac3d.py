import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *

class JAC3D_Test:
    """Test analytical jacobian against autograd"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
    def test(self):
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['z0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        self.thetagt[2] = np.random.normal(0,100)
        iso3d = Iso3D(self.thetagt,self.setup_params)
        dfcs_params = [self.setup_params['zmin'],self.setup_params['alpha'],self.setup_params['beta']]
        adu = iso3d.generate(plot=True)
        jac = jaciso3d(self.thetagt,adu,self.cmos_params,dfcs_params)
        jac_auto = jaciso_auto3d(self.thetagt,adu,self.cmos_params,dfcs_params)
        print(jac-jac_auto)
