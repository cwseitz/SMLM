from SMLM.utils.pipes import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230324_Hela_ buffer_j646_50pm overnight_high power_no405_50ms_02-sub'
]

with open('storm2d.json', 'r') as f:
    config = json.load(f)
with open('setup2d.json', 'r') as f:
    setup = json.load(f)
    
for prefix in prefixes:
    print("Processing " + prefix)
    path = config['analpath']+prefix+'/'+prefix+'_spots.csv'
    spots = pd.read_csv(path)
    fig,ax=plt.subplots()
    ax.scatter(spots['y'],spots['x'],color='cornflowerblue',marker='x',s=1)
    plt.show()

