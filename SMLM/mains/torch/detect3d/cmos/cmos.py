import numpy as np

gain = 2.2*np.ones((121,121))
offset = 100*np.ones((121,121))
var = 5*np.ones((121,121))

np.savez('gain.npz',gain)
np.savez('offset.npz',offset)
np.savez('var.npz',var)
