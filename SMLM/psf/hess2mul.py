import numpy as np
import matplotlib.pyplot as plt
from .hess2 import hessian2

def hess2mul(x,y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    hessblock = [hessian2(x,y,*theta[:,n],eta,texp,gain,var) for n in range(nspots)]
    hessblock = np.array(hessblock)
    nspots,ntheta,_,L,_ = hessblock.shape
    out = np.zeros(nspots*ntheta,nspots*ntheta,L,L)
    for n in range(L):
        for m in range(L):
            this_hess = hessblock[:,:,:,n,m]
    hessblock = hessblock.reshape(nspots*ntheta,nspots*ntheta,L,L)
    plt.imshow(hessblock[:,:,0,0])
    plt.show()
    return hessblock
