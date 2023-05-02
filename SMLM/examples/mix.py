import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.psf2d import *
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

nspots = 5
theta0 = np.zeros((4,nspots))
theta0[0,:] = np.random.normal(L/2,2.0,size=nspots)
theta0[1,:] = np.random.normal(L/2,2.0,size=nspots)
theta0[2,:] = sigma
theta0[3,:] = N0

frame = FrameMix(theta0,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)

fig, ax = plt.subplots(1,2)
hess = hessmix_auto(theta0,adu,cmos_params)
ax[0].imshow(hess,cmap='gray')
hess = hessmix(theta0,adu,cmos_params)
ax[1].imshow(hess,cmap='gray')
plt.show()
