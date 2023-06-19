import numpy as np
from SMLM.tests import *
import json

with open('setup2d.json', 'r') as f:
    setup_params = json.load(f)
test = SGLD2D_Test(setup_params)
test.test()

