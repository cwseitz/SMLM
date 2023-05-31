import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import erf

def defocus_func(z0,sigma,zmin,ab):
    pixel_size = 108.3
    z0 = pixel_size*z0
    sigma_x = sigma + ab*(z0+zmin)**2
    sigma_y = sigma + ab*(z0-zmin)**2
    return sigma_x, sigma_y   

