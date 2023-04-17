import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

def cluster(stack,mask):

    nt,nx,ny = stack.shape
    time_series = stack.reshape((nt,nx*ny))
    mask = mask.reshape((nx*ny))
    idx = np.argwhere(mask == 255).squeeze()
    time_series = time_series[:,idx]

    nt, npix = time_series.shape
    corr_matrix = np.corrcoef(time_series.T, rowvar=True)
    Z = linkage(corr_matrix, method='complete', metric='euclidean')
    max_num_clusters = 10
    cluster_ids = fcluster(Z, max_num_clusters, criterion='maxclust')
    
    colored_pixels = np.zeros((npix, 3))
    for i in range(max_num_clusters):
        color = np.random.rand(3)
        colored_pixels[cluster_ids == i, :] = color

    colored_image = np.zeros((nx*ny,3))
    colored_image[idx,:] = colored_pixels
    colored_image = colored_image.reshape((nx,ny,3))
    return colored_image

path = '/research3/shared/cwseitz/Data/STORM/230324_Hela_ buffer_j646_50pm overnight_high power_no405_10ms_01/'

file = '230324_Hela_ buffer_j646_50pm overnight_high power_no405_10ms_01_MMStack_Pos0.ome.tif'

maskfile = '230324_Hela_ buffer_j646_50pm overnight_high power_no405_10ms_01_MMStack_Pos0.ome-mask.tif'

stack = imread(path+file)
mask = imread(path+maskfile)
clustered_image = cluster(stack,mask)
frame = stack[0] 

# Set the transparency of the clustered image
alpha = 0.3

# Display the overlayed image using plt.imshow()
plt.imshow(frame, cmap='gray')
plt.imshow(clustered_image, alpha=alpha)
plt.show()
