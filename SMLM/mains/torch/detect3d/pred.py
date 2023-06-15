from SMLM.tests import CNN3D_Test
from SMLM.generators import Mix3D
from SMLM.torch.train.metrics import jaccard_coeff
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

with open('setup.json', 'r') as f:
    setup_config = json.load(f)
with open('train.json', 'r') as f:
    train_config = json.load(f)
with open('pred.json', 'r') as f:
    pred_config = json.load(f)
    

modelpath = '/home/cwseitz/git/SMLM/SMLM/mains/torch/detect3d/saved/models/SMLM/0615_031847/'
modelname='checkpoint-epoch50.pth'
test = CNN3D_Test(setup_config,train_config,pred_config,modelpath,modelname)

generator = Mix3D(setup_config)
for n in range(1):
    sample, target, theta = generator.generate()
    xyz_pred = test.forward(sample,show=True)
    xyz_pred = xyz_pred.astype(np.float32)
    xyz_pred /= 4.0
    plt.imshow(sample.squeeze())
    plt.scatter(xyz_pred[:,1],xyz_pred[:,0],color='red',marker='x')
    plt.scatter(theta[1,:],theta[0,:],color='blue',marker='x')
    plt.show()


    

