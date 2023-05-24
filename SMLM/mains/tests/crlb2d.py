import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf2d import crlb2d
from SMLM.torch.pred import *
from numpy.random import beta
from scipy.stats import multivariate_normal

#########################
# Parameters
#########################

L = 20
omat = np.ones((L,L))
gain0 = 1.0 #huang 2013
offset0 = 0.0
var0 = 100.0
gain = gain0*omat #ADU/e-
offset = offset0*omat #ADU
var = var0*omat #ADU^2
pixel_size = 108.3 #nm
sigma = 0.22*640/1.4 #zhang 2007
sigma = sigma = sigma/pixel_size
texp = 1.0 #seconds
eta = 1.0 #quantum efficiency
N0 = 1000
cmos_params = [eta,texp,gain,var]

thetagt = np.zeros((4,))
thetagt[0] = 5
thetagt[1] = 10
thetagt[2] = sigma
thetagt[3] = N0

frame = Iso2D(thetagt,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)
crlb = crlb2d(thetagt,adu,*cmos_params)
print(np.sqrt(crlb))
