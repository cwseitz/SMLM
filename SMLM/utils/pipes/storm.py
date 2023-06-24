import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from SMLM.utils.localize import LoGDetector
from SMLM.utils import RLDeconvolver
from SMLM.psf.psf2d import MLEOptimizer2DGrad
from skimage.filters import gaussian

class PipelineMLE2D:
    def __init__(self,config,prefix):
        self.config = config
        self.prefix = prefix
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.stack = tifffile.imread(self.datapath+self.prefix+'.tif')
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
    def localize(self,plot=False):
        path = self.analpath+self.prefix+'/'+self.prefix+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        deconv = RLDeconvolver()
        threshold = self.config['threshold']
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = deconv.deconvolve(self.stack[n])
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect()
                log.show(self.stack[n])
                plt.show()
        else:
            print('Spot files exist. Skipping')
    def fit(self,plot=False):
        opt = MLEOptimizer2DGrad()
        
class PipelineLifetime:
    def __init__(self,config,prefix,dt=10,plot=True):
        self.config = config
        self.prefix = prefix
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.stack = tifffile.imread(self.datapath+self.prefix+'.tif')
        deconv = RLDeconvolver()
        threshold = self.config['threshold']
        framed = deconv.deconvolve(self.stack[0])
        log = LoGDetector(framed,threshold=threshold)
        spots = log.detect(); log.show(self.stack[0]); plt.show()
        xidx = spots['x'].to_numpy().astype(np.int16)
        yidx = spots['y'].to_numpy().astype(np.int16)
        vals = self.stack[:,xidx,yidx]
        nt,nspots = vals.shape
        thresh = np.mean(vals,axis=0)
        thresh_vals = vals > thresh.reshape(1, -1) 
        binary = thresh_vals.astype(int)
        all_off_times = []; all_on_times = []
        for n in range(nspots):
            off_times, on_times = self.lifetime(binary[:,n],dt)
            all_off_times += off_times; all_on_times += on_times
        if plot:
            self.plot(all_off_times,all_on_times,dt)
            
    def plot(self,off_times,on_times,dt):
        mx = max(on_times+off_times); mx = 250
        bins = np.arange(0,mx,dt)        
        fig,ax = plt.subplots(1,2,figsize=(5,3))
        ax[0].hist(off_times,bins=bins,color='black')
        ax[0].set_xlabel('OFF lifetime')
        ax[1].hist(on_times,bins=bins,color='black')
        ax[1].set_xlabel('ON lifetime')
        plt.tight_layout()
        plt.show()

    def lifetime(self,X,dt):
        X1 = X
        X2 = np.logical_not(X)
        off_times = []; on_times = []
        x1 = np.argwhere(X1 == 1).flatten()
        x2 = np.argwhere(X2 == 1).flatten()
        diff1 = np.diff(x1)
        diff2 = np.diff(x2)
        diff1 = diff1[np.argwhere(diff1 >= 2)] - 1
        diff2 = diff2[np.argwhere(diff2 >= 2)] - 1
        off_times += list(np.squeeze(diff2)*dt)
        on_times += list(np.squeeze(diff1)*dt)
        return off_times, on_times

