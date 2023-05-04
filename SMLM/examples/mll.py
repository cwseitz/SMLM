import numpy as np
import matplotlib.pyplot as plt
from SMLM.psf2d import *

#########################
# Parameters
#########################

L = 20
omat = np.ones((L,L))
gain0 = 2.2 #huang 2013
offset0 = 100.0
var0 = 0.0
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

theta0 = np.zeros((4,))
x0 = np.random.normal(L/2,2.0)
y0 = np.random.normal(L/2,2.0)
theta0[0] = x0
theta0[1] = y0
theta0[2] = sigma
theta0[3] = N0
B0 = 0*np.ones_like(omat)

frame = FrameIso(theta0,eta,texp,L,gain,offset,var,B0)
adu = frame.generate(plot=True)
adu = adu - offset
adu = np.clip(adu,0,None)

#########################
# Marginal log-likelihood
#########################

x0space = np.linspace(x0-5,x0+5,1000)
y0space = np.linspace(y0-5,y0+5,1000)
sigspace = np.linspace(0,4*sigma,1000)
N0space = np.arange(N0-500,N0+500,1)

likex0 = np.zeros_like(x0space)
likey0 = np.zeros_like(y0space)
likesig = np.zeros_like(sigspace)
likeN = np.zeros_like(N0space)

for n in range(len(N0space)):
    likex0[n] = isologlike(np.array([x0space[n],y0,sigma,N0]),adu,*cmos_params,B0)
    likey0[n] = isologlike(np.array([x0,y0space[n],sigma,N0]),adu,*cmos_params,B0)
    likesig[n] = isologlike(np.array([x0,y0,sigspace[n],N0]),adu,*cmos_params,B0)
    likeN[n] = isologlike(np.array([x0,y0,sigma,N0space[n]]),adu,*cmos_params,B0)
    
fig, ax = plt.subplots(1,4,figsize=(10,3))


ax[0].plot(x0space,likex0)
ax[0].vlines(x0,*ax[0].get_ylim(),color='red',linestyle='--')
ax[0].set_xlabel(r'$x_{0}$ (pixels)')
ax[0].set_ylabel('Marginal NLL')

ax[1].plot(y0space,likey0)
ax[1].vlines(y0,*ax[1].get_ylim(),color='red',linestyle='--')
ax[1].set_xlabel(r'$y_{0}$ (pixels)')
ax[1].set_ylabel('Marginal NLL')

ax[2].plot(sigspace,likesig)
ax[2].vlines(sigma,*ax[2].get_ylim(),color='red',linestyle='--')
ax[2].set_xlabel(r'$\sigma$ (pixels)')
ax[2].set_ylabel('Marginal NLL')

ax[3].plot(N0space,likeN)
ax[3].vlines(N0,*ax[3].get_ylim(),color='red',linestyle='--')
ax[3].set_xlabel(r'$N_{0}$ (photons)')
ax[3].set_ylabel('Marginal NLL')


plt.tight_layout()
plt.show()
