import numpy as np
from SMLM.tests import *
import json

with open('setup2d.json', 'r') as f:
    setup_params = json.load(f)
mle2dtest = MLE2DGrad_Test(setup_params)
mle2dtest.test()



