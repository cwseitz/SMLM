import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.generators import Iso2D
from SMLM.psf2d import jaciso
from SMLM.torch.pred import *
from numpy.random import beta
from scipy.stats import multivariate_normal

#########################
# Parameters
#########################

L = 20
omat = np.ones((L,L))
gain0 = 2.2 #huang 2013
offset0 = 0.0
var0 = 100.0
gain = gain0*omat #ADU/e-
offset = offset0*omat #ADU
var = var0*omat #ADU^2
pixel_size = 108.3 #nm
sigma = 0.22*640/1.4 #zhang 2007
sigma = sigma = sigma/pixel_size
texp = 1.0 #seconds
eta = 0.8 #quantum efficiency
N0 = 1000
cmos_params = [eta,texp,gain,var]

theta0 = np.zeros((4,1))
theta0[0] = 5.3
theta0[1] = 10.2
theta0[2] = sigma
theta0[3] = N0

frame = Iso2D(theta0,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)

theta0[0] += 1.5
theta0[1] += 1.5
opt = Optimizer(theta0,adu,cmos_params)
theta = opt.optimize()
print(theta)