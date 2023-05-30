import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf3d import jaciso3d, jaciso_auto3d
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
var0 = 1.0
gain = gain0*omat #ADU/e-
offset = offset0*omat #ADU
var = var0*omat #ADU^2
pixel_size = 108.3 #nm
sigma = 0.22*640/1.4 #zhang 2007
sigma = sigma/pixel_size
texp = 1.0 #seconds
eta = 0.8 #quantum efficiency
N0 = 1000
cmos_params = [eta,texp,gain,var]

thetagt = np.zeros((5,))
thetagt[0] = 5.3
thetagt[1] = 10.2
thetagt[2] = 5.0
thetagt[3] = sigma
thetagt[4] = N0

frame = Iso3D(thetagt,eta,texp,L,gain,offset,var,pixel_size)
adu = frame.generate(plot=True)

theta0 = np.zeros_like(thetagt)
theta0[0] = thetagt[0] + np.random.normal(0,1)
theta0[1] = thetagt[1] + np.random.normal(0,1)
theta0[2] = thetagt[2] + np.random.normal(0,1)
theta0[3] = thetagt[3]
theta0[4] = thetagt[4]

lr = np.array([0.001,0.001,0.001,0,0]) #hyperpar
opt = MLEOptimizer3D(theta0,adu,cmos_params)
theta, loglike = opt.optimize(iters=1000,lr=lr)
print(f'Thetagt: {thetagt}')
print(f'Theta: {theta}')
print(f'Diff: {theta-thetagt}')
plt.plot(loglike)
plt.show()


