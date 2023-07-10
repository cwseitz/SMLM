import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
from pathlib import Path
from SMLM.utils.localize import LoGDetector
from SMLM.cluster import VBGMM, LFilter
from SMLM.utils import RLDeconvolver
from SMLM.psf.psf2d import MLEOptimizer2DGrad, SGLDSampler2D, hessiso_auto2d
from skimage.filters import gaussian
from numpy.linalg import inv

class PipelineMLE2D:
    def __init__(self,config,setup,prefix):
        self.config = config
        self.prefix = prefix
        self.setup = setup
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.stack = tifffile.imread(self.datapath+self.prefix+'-sub.tif')
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [setup['nx'],setup['ny'],
                            setup['eta'],setup['texp'],
                            np.load(setup['gain'])['arr_0'],
                            np.load(setup['offset'])['arr_0'],
                            np.load(setup['var'])['arr_0']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.prefix+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot=False,tmax=None):
        path = self.analpath+self.prefix+'/'+self.prefix+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        deconv = RLDeconvolver()
        if tmax is not None: nt = tmax
        threshold = self.config['threshold']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = deconv.deconvolve(self.stack[n],iters=5)
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot:
                    log.show(); plt.show()
                spots = self.fit(framed,spots)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')

    def scatter_samples(self,adu,samples,theta_mle=None,theta_sgld=None):
        fig, ax = plt.subplots()
        ax.imshow(adu,cmap='gray')
        std = np.round(np.std(samples,axis=0),3)
        labelstr = f'sx={std[0]}, sy={std[1]}'
        ax.scatter(samples[:,0],samples[:,1],color='black',label=labelstr)
        if theta_mle is not None:
            ax.scatter(theta_mle[0],theta_mle[1],color='red',label='MLE')
        if theta_sgld is not None:
            ax.scatter(theta_sgld[0],theta_sgld[1],color='blue',label='SGLD')
        ax.legend()
        plt.tight_layout()
        
    def get_errors(self,theta,adu):
        hess = hessiso_auto2d(theta,adu,self.cmos_params)
        try:
            errors = np.sqrt(np.diag(inv(hess)))
        except:
            errors = np.empty((4,))
            errors[:] = np.nan
        return errors
        
    def fit(self,frame,spots,plot=False,patchw=3):
        lr = np.array([0.0001,0.0001,0.0,350.0])
        spots['x_mle'] = None; spots['y_mle'] = None; spots['N0'] = None;
        spots['x_err'] = None; spots['y_err'] = None; spots['s_err'] = None; spots['N0_err'] = None;
        for i in spots.index:
            print(f'Fitting spot {i}')
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[5]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.setup['sigma'],self.setup['N0']])
            opt = MLEOptimizer2DGrad(theta0,adu,self.setup) #cartesian coordinates with top-left origin
            theta_mle, loglike = opt.optimize(iters=20,plot=False,lr=lr)
            error_mle = self.get_errors(theta_mle,adu)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[3]
            spots.at[i, 'x_err'] = error_mle[1]
            spots.at[i, 'y_err'] = error_mle[0]
            spots.at[i, 's_err'] = error_mle[2]
            spots.at[i, 'N0_err'] = error_mle[3]
        return spots
    def save(self,spotst):
        path = self.analpath+self.prefix+'/'+self.prefix+'_spots.csv'
        spotst.to_csv(path)
        
    def scatter(self):
        path = self.analpath+self.prefix+'/'+self.prefix+'_spots.csv'
        spots = pd.read_csv(path)
        fig,ax=plt.subplots()
        spots = spots.loc[spots['N0'] > 0]
        spots = spots.loc[spots['frame'] < 50]
        ax.scatter(spots['x_mle'],spots['y_mle'],s=1,marker='x',color='cornflowerblue')

class PipelineCluster2D:
    def __init__(self,config,setup,prefix):
        self.config = config
        self.prefix = prefix
        self.setup = setup
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.stack = tifffile.imread(self.datapath+self.prefix+'.tif')[:1000]
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [setup['nx'],setup['ny'],
                            setup['eta'],setup['texp'],
                            np.load(setup['gain'])['arr_0'],
                            np.load(setup['offset'])['arr_0'],
                            np.load(setup['var'])['arr_0']]  

    def scatter(self,spots):
        fig,ax=plt.subplots()
        ax.invert_yaxis() #for top-left origin
        ax.scatter(spots['y_mle'],spots['x_mle'],color='cornflowerblue',marker='x',s=1)
        ax.set_aspect('equal')
        plt.show()

    def selectROI(self,max_err=0.03):
        path = self.config['datapath']+'/'+self.prefix
        mask = tifffile.imread(path+'-mask.tif')
        path = self.config['analpath']+self.prefix+'/'+self.prefix
        spots = pd.read_csv(path+'_spots.csv')
        spots = spots.dropna()
        spots = spots.loc[(spots['x_err'] <= max_err) & (spots['y_err'] <= max_err)]
        spots['x'] = spots['x'].astype(int)
        spots['y'] = spots['y'].astype(int)
        spots['mask_value'] = mask[spots['x'],spots['y']]
        spots = spots[spots['mask_value'] > 0]
        self.scatter(spots)
         
    def showROI(self,centers,hw=15,max_err=0.03):
        path = self.config['datapath']+'/'+self.prefix
        mask = tifffile.imread(path+'-mask.tif')
        path = self.config['analpath']+self.prefix+'/'+self.prefix
        spots = pd.read_csv(path+'_spots.csv')
        spots = spots.dropna()
        spots = spots.loc[(spots['x_err'] <= max_err) & (spots['y_err'] <= max_err)]
        spots['x'] = spots['x'].astype(int)
        spots['y'] = spots['y'].astype(int)
        spots['mask_value'] = mask[spots['x'],spots['y']]
        spots = spots[spots['mask_value'] > 0]
        for i,(xr,yr) in enumerate(centers):
            spotsROI = spots.loc[(spots['x'] > xr-hw) & 
                                 (spots['x'] < xr+hw) & 
                                 (spots['y'] > yr-hw) & 
                                 (spots['y'] < yr+hw)]
            self.scatter(spotsROI)

    def clustROI(self,centers,hw=15,max_err=0.03,r=1.0,T=1.0,plot=False):
        path = self.config['datapath']+'/'+self.prefix
        mask = tifffile.imread(path+'-mask.tif')
        path = self.config['analpath']+self.prefix+'/'+self.prefix
        spots = pd.read_csv(path+'_spots.csv')
        spots = spots.dropna()
        spots = spots.loc[(spots['x_err'] <= max_err) & (spots['y_err'] <= max_err)]
        spots['x'] = spots['x'].astype(int)
        spots['y'] = spots['y'].astype(int)
        spots['mask_value'] = mask[spots['x'],spots['y']]
        spots = spots[spots['mask_value'] > 0]
        
        for i,(xr,yr) in enumerate(centers):
            spotsROI = spots.loc[(spots['x'] > xr-hw) & 
                                 (spots['x'] < xr+hw) & 
                                 (spots['y'] > yr-hw) & 
                                 (spots['y'] < yr+hw)]
            xlim = (spotsROI['x'].min(),spotsROI['x'].max())
            ylim = (spotsROI['y'].min(),spotsROI['y'].max())
            X = LFilter(spotsROI[['x_mle','y_mle']].values,xlim,ylim,r,T)
            gmm = VBGMM(X)
            gmm.fit(plot=plot)

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
        thresh_vals = vals > thresh.reshape(1,-1) 
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
        
        


