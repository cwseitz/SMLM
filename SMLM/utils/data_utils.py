import torch
import numpy as np

# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, upsampling_factor, pixel_size_lateral, pixel_size_axial, zhrange, grid_shape):
       
    # current dimensions
    H, W, D = grid_shape
        
    # number of particles
    batch_size, num_particles, nfeatures = xyz_np.shape

    zshift = xyz_np[:,:,2] + zhrange/pixel_size_axial       
    xg = (np.floor(upsampling_factor*xyz_np[:,:,0])).astype('int')
    yg = (np.floor(upsampling_factor*xyz_np[:,:,1])).astype('int')
    zg = (np.floor(zshift)).astype('int')
    
    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()
    
    # update dimensions
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    
    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indX, indY])
    else:
        ibool = torch.LongTensor([indZ, indX, indY])

    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    
    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    else:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([D, H, W])).to_dense()
    return boolean_grid

