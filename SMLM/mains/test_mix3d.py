from SMLM.tests import Mix3D_Test
from SMLM.generators import Mix3D
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

with open('torch/detect3d/setup.json', 'r') as f:
    setup_config = json.load(f)
test = Mix3D_Test(setup_config)
test.forward()

