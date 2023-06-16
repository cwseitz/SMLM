import numpy as np
import json
from SMLM.tests import MLE3D_Test

with open('setup3d.json', 'r') as f:
    setup_params = json.load(f)
    
mle3dtest = MLE3D_Test(setup_params)
mle3dtest.test()

