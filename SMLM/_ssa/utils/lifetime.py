import numpy as np
import matplotlib.pyplot as plt

def bin_lifetime(life0,life1,bins,density=False):
    vals0, bins0 = np.histogram(life0,bins=bins,density=density)
    vals1, bins1 = np.histogram(life1,bins=bins,density=density)
    return vals0, vals1

def lifetime4s(X,dt):
    nn,ns,nt = X.shape
    Xnew = np.zeros((nn,2,nt))
    Xnew[:,0,:] = X[:,0,:]
    Xnew[:,1,:] = np.sum(X[:,1:,:],axis=1)
    X = Xnew
    X1 = X[:,0,:] #on state
    X2 = X[:,1,:] #off state
    nparticles,nt = X1.shape
    times1 = []; times2 = []
    for particle in range(nparticles):
        x1 = np.argwhere(X1[particle] == 1).flatten()
        x2 = np.argwhere(X2[particle] == 1).flatten()
        diff1 = np.diff(x1)
        diff2 = np.diff(x2)
        diff1 = diff1[np.argwhere(diff1 >= 2)] - 1
        diff2 = diff2[np.argwhere(diff2 >= 2)] - 1
        times1 += list(diff2) #swap
        times2 += list(diff1)
    times1 = np.array(times1)*dt
    times2 = np.array(times2)*dt
    return times1, times2
    
    

