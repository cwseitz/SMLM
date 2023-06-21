import numpy as np
import matplotlib.pyplot as plt
import json
from SMLM.generators import Mix3D

with open('setup.json', 'r') as f:
    setup_config = json.load(f)

ntrain = 10
for n in range(ntrain):
    mix3d = Mix3D(setup_config)
    adu,spikes,theta = mix3d.generate()
    plt.imshow(adu[0])
    plt.show()
