import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import convolve2d as conv2
from scipy.ndimage import gaussian_filter
from skimage import color, data, restoration
from skimage import img_as_float
from skimage.filters import median, gaussian
from skimage.io import imread, imsave

def gkern(l=5, sig=0.92):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

path = '/research3/shared/cwseitz/Data/STORM/'
file = '230516_Hela_j646_50pm overnight_High_10ms_10000frames_buffer_03-sub'
stack = tifffile.imread(path+file+'.tif')
deconv = np.zeros_like(stack)
psf = gkern()

for i,frame in enumerate(stack):
    print(f'Deconvolving frame {i}/10000')
    frame = img_as_float(frame)
    deconv[i] = restoration.richardson_lucy(frame, psf, num_iter=5)
    
imsave(path+file+'-rl.tif',deconv)
