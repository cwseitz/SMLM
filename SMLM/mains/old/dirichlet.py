import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.psf import *
from SMLM.localize import *
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
B0 = 20

nspots = 3
theta0 = np.zeros((5,nspots))
theta0[0,:] = np.random.normal(L/2,2.0,size=nspots)
theta0[1,:] = np.random.normal(L/2,2.0,size=nspots)
theta0[2,:] = sigma
theta0[3,:] = N0
theta0[4,:] = B0

#########################
# MCMC
#########################

frame = Frame(theta0,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)

#detector = LOGDetector(adu,threshold=50.0)
#spots = detector.detect()
#detector.show(); plt.show()
xprior = L/2; yprior = L/2

n0r = 1000
niter = 10
nchains = 1
beta = 1e-3

np.savez('/home/cwseitz/Desktop/chains/theta0.npz',theta0)
dp = DeconDP(adu,eta,texp,gain,var)
prior_params = (xprior,yprior,sigma,n0r,B0)
start = time.time()
for n in range(nchains):
    thetat = dp.run_mcmc(prior_params,niter,beta=beta,show=False)
    chain_id = str(uuid.uuid4())
    thetat = thetat.assign(chain_id=chain_id)
    thetat.to_csv('/home/cwseitz/Desktop/chains/' + chain_id + '.csv')
end = time.time()
runtime = end - start
print(f'MCMC Runtime: {runtime} sec')
print(thetat)

