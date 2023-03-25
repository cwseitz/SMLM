import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..models import Gaussian2D

class GaussianPSFFitter:

    def __init__(self,X,blobs_df,diagnostic=False):
        self.X = X
        self.blobs_df = blobs_df
        self.diagnostic = diagnostic

    def fit(self):
        df = pd.DataFrame([], columns=['frame', 'x0', 'y0', 'A', 'x', 'y', 'sigma'])
        df['r'] = self.blobs_df['r'].to_numpy()
        df['x0'] = self.blobs_df['x'].to_numpy()
        df['y0'] = self.blobs_df['y'].to_numpy()
        for i in df.index:
            x0 = int(df.at[i, 'x0'])
            y0 = int(df.at[i, 'y0'])
            self.delta = int(round(df.at[i, 'r']))
            X = self.X[x0-self.delta:x0+self.delta+1, y0-self.delta:y0+self.delta+1]
            self.model = Gaussian2D()
            self.theta = self.model.fit(X)
            print(self.theta)
            df.at[i, 'A'] = self.theta[0]
            df.at[i, 'x'] = x0 - self.delta + self.theta[1]
            df.at[i, 'y'] = y0 - self.delta + self.theta[2]
            df.at[i, 'sigma'] = self.theta[3]
            if self.diagnostic:
                self.show(X)
        return df

    def show(self,X):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(X, cmap="gray")
        ax.scatter(self.delta,self.delta,s=100,marker='x',c='blue')
        ax.scatter(self.theta[2],self.theta[1],s=100,marker='x',c='red') #cartesian not image coords
        fit = self.model.gaussian(*self.theta)
        plt.contour(fit(*np.indices(X.shape)), cmap='coolwarm')
        plt.show()
