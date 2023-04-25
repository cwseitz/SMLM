import numpy as np
from jac1 import jacobian1

def jac1mul(x,y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    jacblock = [jacobian1(x,y,*theta[:,n],eta,texp,gain,var) for n in range(nspots)]
    return np.concatenate(jacblock)
