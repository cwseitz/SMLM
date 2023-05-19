from SMLM.torch.pred import SynModel
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

modelpath = '/home/cwseitz/Desktop/models/0519_013507/'
datapath = '/home/cwseitz/Desktop/data/Sim_goyuq2uh.tif'
stack = imread(datapath)
nt,nx,ny = stack.shape
modelname='checkpoint-epoch50.pth'
model = SynModel(modelpath,modelname)
mask = model.apply(stack)
imsave('/home/cwseitz/Desktop/test.tif',mask)
