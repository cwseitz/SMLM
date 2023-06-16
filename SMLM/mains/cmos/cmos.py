import numpy as np

gain = 2.2*np.ones((20,20))
offset = 100*np.ones((20,20))
var = 5*np.ones((20,20))

np.savez('gain.npz',gain)
np.savez('offset.npz',offset)
np.savez('var.npz',var)
