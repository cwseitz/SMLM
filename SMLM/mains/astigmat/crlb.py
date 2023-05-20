import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from SMLM.psf3d import *
from numpy.random import beta
from scipy.stats import multivariate_normal

#########################
# Parameters
#########################

L = 20
omat = np.ones((L,L))
gain0 = 2.2 #huang 2013
offset0 = 10.0
var0 = 100.0
gain = gain0*omat #ADU/e-
offset = offset0*omat #ADU
var = var0*omat #ADU^2
pixel_size = 108.3 #nm
sigma = 0.22*640/1.4 #zhang 2007
sigma = sigma = sigma/pixel_size
texp = 1.0 #seconds
eta = 0.8 #quantum efficiency
N0 = 1000 #photons
cmos_params = [eta,texp,gain,var]

theta0 = np.zeros((5,))
theta0[0] = L/2
theta0[1] = L/2
theta0[2] = 500 #nm
theta0[3] = sigma
theta0[4] = N0

#########################
# CRLB
#########################

Nspace = np.linspace(1000/texp,2000/texp,100)
err_arr = np.zeros((4,100))

for j,n0 in enumerate(Nspace):
    theta0[4] = n0
    psf = Astigmatism3D(theta0,eta,texp,L,gain,offset,var)
    adu = psf.generate(plot=False)
    H = hess(theta0,adu,eta,texp,gain,var)
    Hinv = np.linalg.inv(H)
    err = np.sqrt(np.diag(Hinv))
    err_arr[:,j] = err

#########################
# Plot
#########################

fig, ax = plt.subplots(1,3,figsize=(7,2))
colors = ['red','blue','purple']
ax[0].plot(Nspace,err_arr[0,:]*pixel_size,color=colors[0])
ax[1].plot(Nspace,err_arr[1,:]*pixel_size,color=colors[1])
ax[2].plot(Nspace,err_arr[2,:],color=colors[2])
for axi in ax:
    axi.set_xlabel(r'$N_{0}$')
ax[0].set_ylabel(r'$\Delta x$ (nm)')
ax[1].set_ylabel(r'$\Delta y$ (nm)')
ax[2].set_ylabel(r'$\Delta z$ (nm)')
plt.tight_layout()
ax[0].legend()
plt.show()
