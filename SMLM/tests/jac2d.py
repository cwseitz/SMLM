import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf.psf2d import *

class JAC2D_Test:
    """Test analytical jacobian against autograd"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])  
    def test(self):
        iso2d = Iso2D(self.thetagt,self.setup_params)
        adu = iso2d.generate(plot=True)
        jac = jaciso2d(self.thetagt,adu,self.cmos_params)
        jac_auto = jaciso_auto2d(self.thetagt,adu,self.cmos_params)
        print(jac-jac_auto)
