import numpy as np
import matplotlib.pyplot as plt

def bin_ssa(t, x1, x2, x3, x4, dt, T):
    # compute the first difference of the species counts
    dx1 = np.diff(x1, prepend=0)
    dx2 = np.diff(x2, prepend=0)
    dx3 = np.diff(x3, prepend=0)
    dx4 = np.diff(x4, prepend=0)

    # define the time domain for the binned data
    t_bins = np.arange(0, T, dt)
    x1_binned = np.zeros_like(t_bins)
    x2_binned = np.zeros_like(t_bins)
    x3_binned = np.zeros_like(t_bins)
    x4_binned = np.zeros_like(t_bins)

    # assign reaction times to time bins
    bin_indices = np.searchsorted(t_bins, t)
    bin_indices = np.clip(bin_indices - 1, 0, len(t_bins) - 1) # subtract one and clip to valid indices
    x1_binned[bin_indices] = dx1
    x2_binned[bin_indices] = dx2
    x3_binned[bin_indices] = dx3
    x4_binned[bin_indices] = dx4
    
    # accumulate species counts over time
    x1_binned = np.cumsum(x1_binned, axis=0)
    x2_binned = np.cumsum(x2_binned, axis=0)
    x3_binned = np.cumsum(x3_binned, axis=0)
    x4_binned = np.cumsum(x4_binned, axis=0)

    return t_bins, x1_binned, x2_binned, x3_binned, x4_binned
    

def bin_lifetime(life0,life1,bins,density=False):
    vals0, bins0 = np.histogram(life0,bins=bins,density=density)
    vals1, bins1 = np.histogram(life1,bins=bins,density=density)
    return vals0, vals1

def lifetime2s(X,dt):
    """Gets lifetimes for lumped 2-state system"""
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
    
def lifetime4s(X,dt):
    """Gets lifetimes for full 4-state system"""
    ns,nt = X.shape
    times = []
    for i in range(ns):
        times.append([])
        counter = 0
        for j in range(nt):
            state = X[i,j]
            if state == 0:
                times[i].append(counter)
                counter = 0
            else:
                counter += dt
    return times
            

