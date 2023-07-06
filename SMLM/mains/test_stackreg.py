import tifffile
from pystackreg import StackReg
from skimage.io import imsave

sr = StackReg(StackReg.RIGID_BODY)
stack = tifffile.imread('Register.tif')
out = sr.register_transform_stack(stack, reference='first')
imsave('Registered.tif',out)
