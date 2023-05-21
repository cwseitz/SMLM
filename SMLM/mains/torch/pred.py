from SMLM.torch.pred import SynModel
from SMLM.psf2d import mixloglike, jacmix_auto
from skimage.io import imread, imsave
import json
import matplotlib.pyplot as plt
import numpy as np

with open('fast.json', 'r') as f:
    config = json.load(f)
    
modelpath = '/home/cwseitz/Desktop/models/0519_013507/'
datapath = '/home/cwseitz/Desktop/data/Sim_goyuq2uh.tif'
stack = imread(datapath)
nt,nx,ny = stack.shape
modelname='checkpoint-epoch50.pth'
model = SynModel(modelpath,modelname)
mask = model.apply(stack)

gain = np.load(config['gain'])['arr_0']
offset = np.load(config['offset'])['arr_0']
var = np.load(config['var'])['arr_0']
eta = config['eta']
texp = config['texp']
N0 = config['N0']
sigma = config['sigma']
cmos_params = [eta,texp,gain,var]
        
for n in range(nt):
    this_mask = mask[n,0,:,:]
    adu = stack[n,:,:]
    idx = np.argwhere(this_mask > 0)
    nspots,ncoord = idx.shape
    theta = idx.astype(np.float32)
    #n0vec = N0*np.ones((nspots,1))
    #s0vec = sigma*np.ones((nspots,1))
    #theta = np.concatenate((idx,n0vec),axis=1)
    #theta = np.concatenate((theta,s0vec),axis=1)
    jac = jacmix_auto(theta,N0,sigma,adu,cmos_params)
    print(jac)
    #print(nspots,jac.shape)
    

