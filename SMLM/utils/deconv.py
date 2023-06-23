import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import restoration
from skimage import img_as_float

class RLDeconvolver:
    def __init__(self):
        pass
    def gkern(self, l=5, sig=0.92):
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    def deconvolve(self,frame,num_iter=5):
        psf = self.gkern()
        frame = img_as_float(frame)
        deconv = restoration.richardson_lucy(frame, psf, num_iter=num_iter)
        return deconv
        

