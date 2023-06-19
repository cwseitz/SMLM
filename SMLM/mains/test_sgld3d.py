import numpy as np
import json
from SMLM.tests import SGLD3D_Test

with open('setup3d.json', 'r') as f:
    setup_params = json.load(f)
    
sgld3dtest = SGLD3D_Test(setup_params)
sgld3dtest.test()

