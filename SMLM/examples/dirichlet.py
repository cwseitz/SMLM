import numpy as np
import time
import matplotlib.pyplot as plt
from SMLM.psf import *
from SMLM.localize import *
from numpy.random import beta
from scipy.stats import multivariate_normal

#########################
# Parameters
#########################

L = 50
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
B0 = 20

nspots = 50
theta0 = np.zeros((5,nspots))
theta0[0,:] = np.random.uniform(0,L,nspots)
theta0[1,:] = np.random.uniform(0,L,nspots)
theta0[2,:] = sigma
theta0[3,:] = N0
theta0[4,:] = B0

#########################
# MCMC
#########################

frame = Frame(theta0,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)
detector = LOGDetector(adu,threshold=50.0)
spots = detector.detect()
xvec = spots['x'].to_numpy()
yvec = spots['y'].to_numpy() 
n0r = 1000
niter = 1000

dp = DeconDP(adu,eta,texp,gain,var)
prior_params = (xvec,yvec,sigma,n0r,B0)
start = time.time()
thetat = dp.run_mcmc(prior_params,niter,show=False)
end = time.time()
runtime = end - start
print(f'MCMC Runtime: {runtime}')
print(thetat)

