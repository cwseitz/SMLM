import numpy as np

from numpy import sqrt

from numpy import exp

from numpy import pi

from scipy.special import erf


def jacobian1(x, y, x0, y0, z0, sigma, N0, eta, texp, gain, var):
    j_x0 = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))) + erf(sqrt(2)*(y - y0 + 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))))
    j_y0 = N0*eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))))
    j_z0 = N0*eta*gain*texp*(-sqrt(2)*(0.00048 - 1.2e-6*z0)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)**2) + sqrt(2)*(0.00048 - 1.2e-6*z0)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)**2))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2)))) + N0*eta*gain*texp*(-0.25*sqrt(2)*(-1.2e-6*z0 - 0.00048)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)**2) + 0.25*sqrt(2)*(-1.2e-6*z0 - 0.00048)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)**2))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))) + erf(sqrt(2)*(y - y0 + 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))))
    j_s0 = N0*eta*gain*texp*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 - 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 - 400)**2)**2))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2)))) + N0*eta*gain*texp*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*(sigma + 6.0e-7*(z0 + 400)**2)**2))/(sqrt(pi)*(sigma + 6.0e-7*(z0 + 400)**2)**2))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))) + erf(sqrt(2)*(y - y0 + 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))))
    j_N0 = eta*gain*texp*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))) + erf(sqrt(2)*(y - y0 + 0.5)/(2*(sigma + 6.0e-7*(z0 - 400)**2))))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*(sigma + 6.0e-7*(z0 + 400)**2))))
    jac = np.array([j_x0, j_y0, j_z0, j_s0, j_N0], dtype=np.float64)
    return jac
