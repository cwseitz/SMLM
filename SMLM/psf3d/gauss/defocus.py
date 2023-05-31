import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf

def defocus_func(z0,sigma):
    pixel_size = 108.3
    z0 = pixel_size*z0
    zmin = 413.741
    a = 5.349139e-7
    b = 6.016703e-7
    sigma_x = sigma + a*(z0+zmin)**2
    sigma_y = sigma + b*(z0-zmin)**2
    return sigma_x, sigma_y   

