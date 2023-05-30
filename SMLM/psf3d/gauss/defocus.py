import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf

def defocus_func1(z0,sigma):
    sigma_x = sigma + 5.349139e-7*(z0+413.741)**2
    sigma_y = sigma + 6.016703e-7*(z0-413.741)**2
    return sigma_x, sigma_y    

def defocus_func2(z0,sigma):
    pixel_size = 108.3
    zmin = 413.741/pixel_size
    a = 5.349139e-7*pixel_size**2
    b = 6.016703e-7*pixel_size**2
    sigma_x = sigma + a*(z0+zmin)**2
    sigma_y = sigma + b*(z0-zmin)**2
    return sigma_x, sigma_y   

