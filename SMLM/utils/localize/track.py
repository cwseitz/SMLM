import numpy as np
import pandas as pd
import trackpy as tp

class NNTracker:

    def __init__(self,spots,search_range=3,memory=5,min_traj_length=5):
        self.spots = spots
        self.search_range = search_range
        self.memory = memory
        self.min_traj_length = min_traj_length
    def track(self):
        self.spots = self.spots.dropna(subset=['x', 'y', 'frame'])
        self.spots = tp.link_df(self.spots, search_range=self.search_range, memory=self.memory)
        self.spots = tp.filter_stubs(self.spots, self.min_traj_length)
        self.spots = self.spots.reset_index(drop=True)
        self.spots = self.spots.sort_values(['particle', 'frame'])
        self.spots = self.spots.apply(pd.to_numeric)
        return self.spots

