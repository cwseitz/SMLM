from SMLM.psf2d import *
from SMLM.stats import PoissonNormal, PoissonNormalApproximate
from scipy.stats import multivariate_normal
import numpy as np

mu_norm = 100 #ADU
mu_psn = 1000 #e-
sigma_norm = 0.1 #ADU
gain = 2.2 #ADU/e-
pnorm = PoissonNormal(mu_norm,mu_psn,sigma_norm)
pnorm_approx = PoissonNormalApproximate(mu_norm,mu_psn,sigma_norm)
w = 1000

x = np.arange(mu_psn-w,mu_psn+w,10)
pmf = pnorm.get_pmf(x)
pmf_approx = pnorm_approx.get_pmf(x-mu_norm)

samples = pnorm.sample(50000)
samples = np.rint(samples)

vals, bins = np.histogram(samples,bins=x,density=False)
vals = vals/np.sum(vals)

markerline, stemlines, baseline = plt.stem(x,pmf,markerfmt='D',linefmt='red')
markerline.set_markerfacecolor('none')

markerline, stemlines, baseline = plt.stem(x,pmf_approx,linefmt='blue')
markerline.set_markerfacecolor('none')

markerline, stemlines, baseline = plt.stem(bins[:-1],vals,linefmt='black')
markerline.set_markerfacecolor('none')

w = 200
plt.xlabel('ADU')
plt.xlim([mu_psn+mu_norm-w,mu_psn+mu_norm+w])
plt.show()
