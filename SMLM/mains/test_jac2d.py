from SMLM.tests import JAC2D_Test
import matplotlib.pyplot as plt
import json

with open('setup2d.json', 'r') as f:
    setup_params = json.load(f)
jac2dtest = JAC2D_Test(setup_params)
jac2dtest.test()

