import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from SMLM.utils.localize import LoGDetector
from SMLM.utils import RLDeconvolver
from SMLM.psf.psf2d import MLEOptimizer2DGrad

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
