import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from .hess2 import hessian2

def hess2mul(x,y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    blocks = [hessian2(x,y,*theta[:,n],eta,texp,gain,var) for n in range(nspots)]
    blocks = np.array(blocks)
    nspots,ntheta,_,L,_ = blocks.shape
    out = np.zeros((nspots*ntheta,nspots*ntheta,L,L))
    for n in range(L):
        for m in range(L):
            x = list(blocks[:,:,:,n,m])
            out[:,:,n,m] = block_diag(*x)
    return out
