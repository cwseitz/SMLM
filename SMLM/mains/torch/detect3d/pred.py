from SMLM.tests import CNN3D_Test
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

modelpath = '/home/cwseitz/Desktop/Torch/MedZRes/Models/'
modelname='checkpoint-epoch10.pth'
test = CNN3D_Test(setup_config,train_config,pred_config,modelpath,modelname)
xyz_true_batch,xyz_pred_batch = test.test(100,show=True)
z_true_batch = xyz_true_batch[:,2]
z_pred_batch = xyz_pred_batch[:,2]
bin_means,bin_variances  = test.get_errors(z_true_batch,z_pred_batch)

fig,ax=plt.subplots(1,3)
ax[0].scatter(z_true_batch,z_pred_batch,color='black')
ax[1].plot(bin_means,color='red')
ax[2].plot(np.sqrt(bin_variances),color='red')
plt.tight_layout()
plt.show()

