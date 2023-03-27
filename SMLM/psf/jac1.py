import numpy as np

from numpy import sqrt

from numpy import exp

from numpy import pi

from scipy.special import erf


def jacobian1(x, y, x0, y0, sigma, N0, B0, eta, texp, gain, var):
    j_x0 = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))
    j_y0 = N0*eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    j_sigma = eta*gain*texp*(N0*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma))) + N0*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma))))
    j_N0 = eta*gain*texp*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))
    j_B0 = eta*gain*texp
    jac = np.array([j_x0, j_y0, j_sigma, j_N0, j_B0], dtype=np.float64)
    return jac
