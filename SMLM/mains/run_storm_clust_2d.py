from SMLM.utils.pipes import *
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = ['230707_Hela-fixed_j646_50pm overnight_20mW_2000frames_buffer-2-h2b']

with open('storm2d.json', 'r') as f:
    config = json.load(f)
with open('setup2d.json', 'r') as f:
    setup = json.load(f)

ROI = [(165,165),(165,245),(220,180)]    
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = PipelineCluster2D(config,setup,prefix)
    pipe.selectROI()
    #pipe.clustROI(ROI,plot=True)






