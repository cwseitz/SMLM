from SMLM.torch.pred import NeuralEstimator3D
from SMLM.generators import Mix3D
import json
import matplotlib.pyplot as plt
import numpy as np

with open('generate.json', 'r') as f:
    generate_config = json.load(f)
with open('train.json', 'r') as f:
    train_config = json.load(f)

args = train_config['arch']['args']
modelpath = '/home/cwseitz/git/SMLM/SMLM/mains/torch/detect3d/saved/models/SMLM/0613_234439/'
modelname='checkpoint-epoch2.pth'
estimator = NeuralEstimator3D(args['nz'],args['scaling_factor'],args['dilation_flag'],modelpath,modelname)

generator = Mix3D(generate_config)
for n in range(3):
    sample, target = generator.generate()
    output = estimator.forward(sample)
    



    

