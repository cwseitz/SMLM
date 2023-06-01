import numpy as np

gain = 2.2*np.ones((256,256))
offset = 10*np.ones((256,256))
var = 500*np.ones((256,256))

np.savez('gain.npz',gain)
np.savez('offset.npz',offset)
np.savez('var.npz',var)
