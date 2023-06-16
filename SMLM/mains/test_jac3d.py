from SMLM.tests import JAC3D_Test
import matplotlib.pyplot as plt
import json

with open('setup3d.json', 'r') as f:
    setup_params = json.load(f)
    
jac3dtest = JAC3D_Test(setup_params)
jac3dtest.test()

