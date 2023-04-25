import numpy as np
from hess2 import hessian2

def hess2mul(x,y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    hessblock = [hessian2(x,y,*theta[:,n],eta,texp,gain,var) for n in range(nspots)]
    return np.diag(hessblock,k=0)
