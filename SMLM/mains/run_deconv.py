import tifffile
import json
import numpy as np
import matplotlib.pyplot as plt
from SMLM.utils import RLDeconvolver
from skimage.io import imsave

prefixes = [
'230516_Hela_j646_50pm overnight_High_10ms_10000frames_buffer_03-sub'
]

with open('storm2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    stack = tifffile.imread(config['datapath']+prefix+'.tif')
    nt,nx,ny = stack.shape
    stackd = np.zeros((nt,nx,ny),dtype=np.int16)
    deconv = RLDeconvolver()
    for n in range(nt):
        print(f'Deconvolving frame {n}')
        framed = deconv.deconvolve(stack[n])
        stackd[n] = framed
    imsave(config['datapath']+prefix+'-rl.tif',stackd)

