import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial
from .ill_auto import *

def hessiso_auto(theta,adu,eta,texp,gain,var):
    isologlike = isologlike_auto(adu,eta,texp,gain,var)
    hessian_ = hessian(isologlike)
    hess = hessian_(theta)
    return hess




