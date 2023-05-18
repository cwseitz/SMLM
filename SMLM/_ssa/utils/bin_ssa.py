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
