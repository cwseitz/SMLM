from SMLM.torch.pred import NeuralEstimator3D
from SMLM.generators import Mix3D
import json
import matplotlib.pyplot as plt
import numpy as np

with open('setup.json', 'r') as f:
    setup_config = json.load(f)
with open('train.json', 'r') as f:
    train_config = json.load(f)
with open('pred.json', 'r') as f:
    pred_config = json.load(f)
    

modelpath = '/home/cwseitz/git/SMLM/SMLM/mains/torch/detect3d/saved/models/SMLM/0614_041800/'
modelname='checkpoint-epoch10.pth'
estimator = NeuralEstimator3D(setup_config,train_config,pred_config,modelpath,modelname)

generator = Mix3D(setup_config)
for n in range(3):
    sample, target = generator.generate()
    xyz, conf = estimator.forward(sample)
    print(xyz)



    

