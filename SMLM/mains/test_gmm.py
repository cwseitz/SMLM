from SMLM.cluster import *
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefix = '230516_Hela_j646_50pm overnight_High_10ms_10000frames_buffer_02'


with open('storm2d.json', 'r') as f:
    config = json.load(f)
print("Processing " + prefix)


ROI = [(165,165),(165,245),(220,180)]

##################################
# Show whole FOV for ROI selection
##################################

path = config['analpath']+prefix+'/'+prefix
spots = pd.read_csv(path+'-sub_spots.csv')
spots = spots.dropna()
spots = spots.loc[(spots['x_err'] < 0.03) & (spots['y_err'] < 0.03)]

path = config['datapath']+'/'+prefix
mask = imread(path+'-mask.tif')

spots['x'] = spots['x'].astype(int)
spots['y'] = spots['y'].astype(int)
spots['mask_value'] = mask[spots['x'],spots['y']]
spots = spots[spots['mask_value'] > 0]

#fig,ax=plt.subplots()
#ax.invert_yaxis() #for top-left origin
#ax.scatter(spots['y_mle'],spots['x_mle'],color='cornflowerblue',marker='x',s=1) #image coordinates
#ax.set_aspect('equal')
#plt.show()

##################################
# Fit spots with VBGMM
##################################

r = 1.0
T = 1.0

#rseq = np.linspace(1.0,2.0,20)
#thseq = np.linspace(0.1,3.0,20)

hw = 15
for i,(xr,yr) in enumerate(ROI):
    spotsROI = spots.loc[(spots['x'] > xr-hw) & (spots['x'] < xr+hw) & (spots['y'] > yr-hw) & (spots['y'] < yr+hw)]
    xlim = (spotsROI['x'].min(),spotsROI['x'].max())
    ylim = (spotsROI['y'].min(),spotsROI['y'].max())
    X = LFilter(spotsROI[['x_mle','y_mle']].values,xlim,ylim,r,T)
    gmm = VBGMM(X)
    gmm.fit()
    



