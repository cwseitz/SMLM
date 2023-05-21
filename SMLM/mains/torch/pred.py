from SMLM.torch.pred import NeuralEstimator2D
from skimage.io import imread, imsave
import time
import json
import matplotlib.pyplot as plt
import numpy as np

with open('fast.json', 'r') as f:
    config = json.load(f)
    
modelpath = '/home/cwseitz/Desktop/models/0519_013507/'
datapath = '/home/cwseitz/Desktop/data/Sim_goyuq2uh.tif'
gtpath = '/home/cwseitz/Desktop/data/Sim_goyuq2uh_gtmat.npz'

stack = imread(datapath)
modelname='checkpoint-epoch50.pth'
model = NeuralEstimator2D(config,datapath,modelpath,modelname,gtpath=gtpath)
mask = model.forward(stack)




    

