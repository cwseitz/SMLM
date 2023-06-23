import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt

class PipelineSTORM:
    def __init__(self,config,prefix):
        self.config = config
        self.datapath = config['datapath']
        self.analpath = config['analpath']
        self.thresh = config['threshold']
        self.search_range = config['search_range']
        self.memory = config['memory']
        self.min_traj_length = config['min_traj_length']
        self.prefix = prefix
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
    def detect_spots(self,plot=False):
        path = self.analpath+self.prefix+'/'+self.prefix+'_spots.csv'
        file = Path(path)
        if not file.exists():
            print('Running spot detection...')
            detector = DetectorSTORM(self.datapath,self.analpath,self.prefix,self.thresh)
            spotst = detector.detect(plot=plot)
            spotst.to_csv(path)
        else:
            print('Spot files exist. Skipping')
        return spotst
    def track_spots(self,spotst):
        path = self.analpath+self.prefix+'/'+self.prefix+'_tracks.csv'
        file = Path(path)
        if not file.exists():
            print('Tracking spots...')
            tracker = NNTracker(spotst,search_range=self.search_range,
                                       memory=self.memory,
                                       min_traj_length=self.min_traj_length)
            spotst = tracker.track()
            spotst.to_csv(path)
        else:
            print('Track files exist. Skipping')
        return spotst
    def plot_examples(self,nexamples=10,patch_hw=10):
        path = self.analpath+self.prefix+'/'+'examples'
        Path(path).mkdir(parents=True, exist_ok=True)
        path = self.analpath+self.prefix+'/'+self.prefix+'_tracks.csv'
        print('Plotting example tracks...')
        tracks = pd.read_csv(path)
        exs = np.random.choice(tracks['particle'].unique(),size=nexamples,replace=False)
        stack = tifffile.imread(self.datapath+'/'+self.prefix+'/'+self.prefix+'.tif')
        for i,ex in enumerate(exs):
            track = tracks.loc[tracks['particle'] == ex]
            x0,y0 = track[['x','y']].iloc[0].to_numpy()
            x0,y0 = int(x0),int(y0)
            crop = stack[:,x0-patch_hw:x0+patch_hw,y0-patch_hw:y0+patch_hw]
            track['x'] = track['x'] - x0 + patch_hw
            track['y'] = track['y'] - y0 + patch_hw
            anim = anim_blob(track,crop)
            imsave(self.analpath+self.prefix+'/'+'examples/'+f'ex{i}.tif',anim)
        
    def lifetime(self,fps=100):
        T = 1/fps
        path = self.analpath+self.prefix+'/'+self.prefix+'_tracks.csv'
        spots = pd.read_csv(path)
        particles = spots['particle'].unique()
        times = []
        for particle in particles:
            spotsp = spots.loc[spots['particle']==particle]
            frames = spotsp['frame']*T
            diff = np.diff(frames)
            diff = diff[np.where(diff >= 2*T)]
            times += list(diff)
        return np.array(times)       
