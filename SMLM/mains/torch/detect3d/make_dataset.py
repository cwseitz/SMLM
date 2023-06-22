import numpy as np
import uuid
import matplotlib.pyplot as plt
import json
from SMLM.generators import Mix3D
from skimage.io import imsave

with open('setup.json', 'r') as f:
    setup_config = json.load(f)
path = '/home/cwseitz/Desktop/Torch/HighZRes/Train/'
rootname = 'Mix3D_50p_1000N0_400zh_'
ntrain = 1000
for n in range(ntrain):
    mix3d = Mix3D(setup_config)
    adu,spikes,theta = mix3d.generate()
    id = str(uuid.uuid4())
    imsave(path+rootname+id+'.tif',adu)
    np.savez(path+rootname+id+'.npz',spikes=spikes,theta=theta)
    print(spikes.shape)
    del adu,spikes,theta
