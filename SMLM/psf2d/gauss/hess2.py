import numpy as np

from numpy import sqrt

from numpy import exp

from numpy import pi

from scipy.special import erf


def hessian2(x, y, x0, y0, sigma, N0, eta, texp, gain, var):
    h_xx = N0*eta*gain*texp*(0.125*sqrt(2)*(-2*x + 2*x0 - 1.0)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3) - 0.125*sqrt(2)*(-2*x + 2*x0 + 1.0)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))
    h_xy = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))
    h_xs = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2)) + N0*eta*gain*texp*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))*(0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) + 0.25*sqrt(2)*(x - x0 - 0.5)**2*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4) - 0.25*sqrt(2)*(x - x0 + 0.5)**2*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4))
    h_xN = eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))
    h_yx = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))
    h_yy = N0*eta*gain*texp*(sqrt(2)*(-2*y + 2*y0 - 1.0)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3) - sqrt(2)*(-2*y + 2*y0 + 1.0)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    h_ys = N0*eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2)) + N0*eta*gain*texp*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))*(sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) + sqrt(2)*(y - y0 - 0.5)**2*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4) - sqrt(2)*(y - y0 + 0.5)**2*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4))
    h_yN = eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    h_sx = N0*eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2)) + N0*eta*gain*texp*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))*(0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) + 0.125*sqrt(2)*(-2*x + 2*x0 - 1.0)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4) - 0.125*sqrt(2)*(-2*x + 2*x0 + 1.0)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**4))
    h_sy = N0*eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2)) + N0*eta*gain*texp*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))*(sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) + sqrt(2)*(-2*y + 2*y0 - 1.0)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**4) - sqrt(2)*(-2*y + 2*y0 + 1.0)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**4))
    h_ss = 2*N0*eta*gain*texp*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2)) + N0*eta*gain*texp*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))*(-2*sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3) + 2*sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3) + sqrt(2)*(y - y0 - 0.5)**3*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**5) - sqrt(2)*(y - y0 + 0.5)**3*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**5)) + N0*eta*gain*texp*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))*(-0.5*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3) + 0.5*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**3) + 0.25*sqrt(2)*(x - x0 - 0.5)**3*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**5) - 0.25*sqrt(2)*(x - x0 + 0.5)**3*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**5))
    h_sN = eta*gain*texp*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma))) + eta*gain*texp*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    h_Nx = eta*gain*texp*(-0.25*sqrt(2)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + 0.25*sqrt(2)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma)))
    h_Ny = eta*gain*texp*(-sqrt(2)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma) + sqrt(2)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    h_Ns = eta*gain*texp*(0.25*sqrt(2)*(x - x0 - 0.5)*exp(-(x - x0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - 0.25*sqrt(2)*(x - x0 + 0.5)*exp(-(x - x0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-erf(sqrt(2)*(y - y0 - 0.5)/(2*sigma)) + erf(sqrt(2)*(y - y0 + 0.5)/(2*sigma))) + eta*gain*texp*(sqrt(2)*(y - y0 - 0.5)*exp(-(y - y0 - 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2) - sqrt(2)*(y - y0 + 0.5)*exp(-(y - y0 + 0.5)**2/(2*sigma**2))/(sqrt(pi)*sigma**2))*(-0.25*erf(sqrt(2)*(x - x0 - 0.5)/(2*sigma)) + 0.25*erf(sqrt(2)*(x - x0 + 0.5)/(2*sigma)))
    h_NN = np.zeros_like(h_xx)
    H = np.array([[h_xx,h_xy,h_xs,h_xN], [h_yx,h_yy,h_ys,h_yN], [h_sx,h_sy,h_ss,h_sN],[h_Nx,h_Ny,h_Ns,h_NN]], dtype=np.float64)
    return H