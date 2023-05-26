import os
import torch
import matplotlib.pyplot as plt

def RestorationLoss(output,target,cmos_params):
    gain,eta,texp,offset,var = cmos_params
    rate = gain*eta*texp*(output[:,0,:,:]+output[:,1,:,:]) - offset + var
    loss = target*torch.log(rate) - rate - target*np.log(target)
    loss = torch.sum(loss,axis=(1,2))
    return torch.mean(loss,axis=0)

