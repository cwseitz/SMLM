import numpy as np
import pandas as pd
import trackpy as tp

class NNTracker:

    def __init__(self,blobs_df,search_range=3,memory=5,pixel_size=0.1083,
                 frame_rate=3.3,divide_num=5,filters=None,do_filter=False):

        self.blobs_df = blobs_df
        self.search_range = search_range
        self.memory = memory
        self.pixel_size = pixel_size
        self.frame_rate = frame_rate
        self.divide_num = divide_num
        self.filters = filters
        self.do_filter = do_filter

    def track(self):

        self.blobs_df = self.blobs_df.dropna(subset=['x', 'y', 'frame'])
        if self.do_filter:
            self.blobs_df = tp.link_df(self.blobs_df, search_range=self.search_range, memory=self.memory)
            self.blobs_df = tp.filter_stubs(self.blobs_df, 5)
            self.blobs_df = self.blobs_df.reset_index(drop=True)
        else:
            self.blobs_df = tp.link_df(self.blobs_df, search_range=self.search_range, memory=self.memory)
            self.blobs_df = self.blobs_df.reset_index(drop=True)

        self.blobs_df = self.blobs_df.sort_values(['particle', 'frame'])
        blobs_df_cut = self.blobs_df[['frame', 'x', 'y', 'particle']]
        blobs_df_cut = blobs_df_cut.apply(pd.to_numeric)
        im = tp.imsd(blobs_df_cut, mpp=self.pixel_size, fps=self.frame_rate, max_lagtime=np.inf)
        self.blobs_df = self.dcoeff(self.blobs_df, im, self.divide_num)
        self.blobs_df = self.blobs_df.apply(pd.to_numeric)

        return self.blobs_df, im

    def dcoeff(self,traj_df,im,divide_num):

        df = traj_df.copy(deep=True)
        n = int(round(len(im.index)/self.divide_num))
        im = im.head(n)
        #get diffusion coefficient of each particle
        particles = im.columns
        for particle in particles:
            # Remove NaN, Remove non-positive value before calculate log()
            msd = im[particle].dropna()
            msd = msd[msd > 0]
            if len(msd) > 2: # Only fit when msd has more than 2 data points
                x = msd.index.values
                y = msd.to_numpy()
                y = y*1e6 #convert to nm
                popt = fit_msd(x, y)
                df.loc[df['particle']==particle, 'D'] = popt[0]
                df.loc[df['particle']==particle, 'alpha'] = popt[1]
        return df
