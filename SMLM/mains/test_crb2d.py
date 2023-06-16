from SMLM.tests import CRB2D_Test1
import matplotlib.pyplot as plt
import json

with open('setup2d.json', 'r') as f:
    setup_params = json.load(f)
crb2dtest = CRB2D_Test1(setup_params)
crb2dtest.plot()
plt.show()
