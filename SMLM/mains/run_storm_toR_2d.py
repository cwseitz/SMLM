from SMLM.utils.pipes import *
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

fig,ax=plt.subplots()
ax.invert_yaxis() #for top-left origin
ax.scatter(spots['y_mle'],spots['x_mle'],color='cornflowerblue',marker='x',s=1) #image coordinates
ax.set_aspect('equal')
plt.show()

##################################
# Save files for each ROI
##################################

hw = 15
for i,(xr,yr) in enumerate(ROI):
    spotsROI = spots.loc[(spots['x'] > xr-hw) & (spots['x'] < xr+hw) & (spots['y'] > yr-hw) & (spots['y'] < yr+hw)]
    fig,ax=plt.subplots()
    ax.invert_yaxis() #for top-left origin
    ax.scatter(spotsROI['y_mle'],spotsROI['x_mle'],color='cornflowerblue',marker='x',s=1) #image coordinates
    ax.set_aspect('equal')
    plt.show()
    spotsROI['sd'] = 108.3*np.sqrt(spotsROI['x_err']**2 + spotsROI['y_err']**2)
    spotsROI = spotsROI[['x_mle','y_mle','sd']]
    spotsROI = spotsROI.rename(columns={'x_mle':'x','y_mle':'y'})
    Path(config['analpath']+prefix+f'/{i+1}').mkdir(parents=True, exist_ok=True)
    #spotsROI.to_csv(config['analpath']+prefix+f'/{i+1}/data.txt',index=False)



