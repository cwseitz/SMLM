from SMLM.tests import CNN3D_Test
from SMLM.generators import Mix3D
from SMLM.torch.train.metrics import jaccard_coeff
from scipy.spatial.distance import cdist
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

def match(xyz_pred,xyz_true,threshold=3):
    distances = cdist(xyz_true[:,:2],xyz_pred[:,:2])
    xyz_pred_matched = []
    xyz_true_matched = []
    for i in range(len(xyz_true)):
        for j in range(len(xyz_pred)):
            if distances[i, j] < threshold:
                xyz_true_matched.append(xyz_true[i])
                xyz_pred_matched.append(xyz_pred[j])
    xyz_pred_matched = np.array(xyz_pred_matched)
    xyz_true_matched = np.array(xyz_true_matched)
    return xyz_pred_matched, xyz_true_matched

def get_errors(xyz_pred_matched, xyz_true_matched):
    xerr = xyz_pred_matched[:,0] - xyz_true_matched[:,0]
    yerr = xyz_pred_matched[:,1] - xyz_true_matched[:,1]
    return xerr, yerr

def show_pred(frame,xyz_pred,xyz_true):
    frame = np.squeeze(frame)
    fig,ax = plt.subplots()
    ax.imshow(frame,cmap='gray')
    ax.scatter(xyz_pred[:,1],xyz_pred[:,0],marker='x',color='red')
    ax.scatter(xyz_true[:,1],xyz_true[:,0],marker='x',color='blue')
    plt.show()
  
with open('setup.json', 'r') as f:
    setup_config = json.load(f)
with open('train.json', 'r') as f:
    train_config = json.load(f)
with open('pred.json', 'r') as f:
    pred_config = json.load(f)
    

modelpath = '/home/cwseitz/git/SMLM/SMLM/mains/torch/detect3d/saved/models/SMLM/0620_002741/'
modelname='checkpoint-epoch50.pth'
test = CNN3D_Test(setup_config,train_config,pred_config,modelpath,modelname)

generator = Mix3D(setup_config)
xerrs = []; yerrs = []
for n in range(1):
    sample, target, theta = generator.generate()
    xyz_true = theta[:3,:].T
    xyz_pred = test.forward(sample,show=True)
    xyz_pred = xyz_pred.astype(np.float32)
    xyz_pred[:,0] = xyz_pred[:,0]/4.0
    xyz_pred[:,1] = xyz_pred[:,1]/4.0
    show_pred(sample,xyz_pred,xyz_true)
    xyz_pred_matched, xyz_true_matched = match(xyz_pred,xyz_true)
    print(xyz_pred_matched)
    print(xyz_true_matched)
    #xerr, yerr = get_errors(xyz_pred_matched, xyz_true_matched)
    #xerrs.append(xerr); yerrs.append(yerr)

#xerrs = np.concatenate(xerrs)
#yerrs = np.concatenate(yerrs)
#fig, ax = plt.subplots(1,2)
#ax[0].hist(xerrs)
#ax[1].hist(yerrs)
#plt.show()

    

