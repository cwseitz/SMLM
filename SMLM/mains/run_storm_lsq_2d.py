from SMLM.utils.pipes import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230516_Hela_j646_50pm overnight_High_10ms_10000frames_buffer_02-sub'
]

with open('storm2d.json', 'r') as f:
    config = json.load(f)
with open('setup2d.json', 'r') as f:
    setup = json.load(f)
    
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = PipelineLSQ2D(config,setup,prefix)
    pipe.localize()
