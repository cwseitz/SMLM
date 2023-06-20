from SMLM.utils import PoissonNormal, PoissonNormalApproximate
from SMLM.generators import Iso2D
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import json

def get_kdist(offset,rate,std,gain=2.2,w=1000,ax=None):
    pnorm = PoissonNormal(offset,rate,std)
    pnorm_approx = PoissonNormalApproximate(offset,rate,std)
    x = np.arange(0,w,1)
    pmf = pnorm.get_pmf(x)
    pmf_approx = pnorm_approx.get_pmf(x-offset)
    cmf = np.cumsum(pmf)
    cmf_approx = np.cumsum(pmf_approx)
    if ax:
        ax.plot(cmf,color='red',linestyle='--',label='cmf')
        ax.plot(cmf_approx,color='purple',label='approx')
        ax.plot(np.abs(cmf-cmf_approx),color='cornflowerblue',label='abs diff')
        ax.set_xlim([offset+rate-50,offset+rate+50])
        ax.legend()
    kdist = np.max(np.abs(cmf-cmf_approx))
    return kdist
