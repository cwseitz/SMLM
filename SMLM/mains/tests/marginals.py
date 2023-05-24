import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

def post_marginals(chainsk,theta0):
    fig, ax = plt.subplots(1,3)
    ntheta, nspots = theta0.shape
    x0s = theta0[0,:]
    y0s = theta0[1,:]
    n0s = theta0[4,:]
    ax[0].hist(chainsk['x0'],bins=20,color='black')
    ax[1].hist(chainsk['y0'],bins=20,color='black')
    ax[2].hist(chainsk['n0'],bins=20,color='black')
    for n in range(nspots):
       ymin0,ymax0 = ax[1].get_ylim()
       ax[0].vlines(theta0[1,n],ymin=ymin0,ymax=ymax0)
    plt.show()

dir = '/home/cwseitz/Desktop/chains/'
files = glob(dir + '*.csv')
chains = [pd.read_csv(file) for file in files]
chains = pd.concat(chains)
chains = chains.loc[chains['iteration'] > 500]
kunique = sorted(chains['K'].unique())
theta0 = np.load(dir+'theta0.npz')['arr_0']
print(theta0)

for k in kunique:
    chainsk = chains.loc[chains['K'] == k]
    print(chainsk)
    post_marginals(chainsk,theta0)
