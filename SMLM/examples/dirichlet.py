import numpy as np
import matplotlib.pyplot as plt
from SMLM.psf import *

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

#########################
# Parameters
#########################

L = 20
omat = np.ones((L,L))
gain0 = 2.2 #huang 2013
offset0 = 0.0
var0 = 100.0
gain = gain0*omat #ADU/e-
offset = offset0*omat #ADU
var = var0*omat #ADU^2
pixel_size = 108.3 #nm
sigma = 0.22*640/1.4 #zhang 2007
sigma = sigma = sigma/pixel_size
texp = 1.0 #seconds
eta = 0.8 #quantum efficiency
N0 = 1000
B0 = 100

nspots = 5
theta0 = np.zeros((5,nspots))
theta0[0,:] = np.random.uniform(0,L,nspots)
theta0[1,:] = np.random.uniform(0,L,nspots)
theta0[2,:] = sigma
theta0[3,:] = N0
theta0[4,:] = B0
frame = Frame(theta0,eta,texp,L,gain,offset,var)
adu = frame.generate(plot=True)

adu = torch.tensor(adu)
gain = torch.tensor(gain)
var = torch.tensor(var)
eta = torch.tensor(eta)
texp = torch.tensor(texp)

#########################
# Sampling algorithm
#########################

# Define the prior distribution on the number of particles using a Dirichlet process
def dirichlet_process(alpha):
    beta = dist.Beta(1, alpha)
    weights = [beta.sample()]
    while sum(weights) < 1:
        weights.append(beta.sample())
    return torch.tensor(weights[:-1])

# Define the Pyro model
def model(adu, eta, texp, gain, var):
    # Sample the number of particles using a Dirichlet process prior
    alpha = 1
    weights = dirichlet_process(alpha)
    nspots = len(weights)
    # Sample the particle parameters from the prior
    sigma = dist.Uniform(0, 5).sample(torch.Size([nspots]))
    x0 = dist.Uniform(0, adu.shape[0]).sample(torch.Size([nspots]))
    y0 = dist.Uniform(0, adu.shape[1]).sample(torch.Size([nspots]))
    N0 = dist.Uniform(0, 2000).sample(torch.Size([nspots]))
    B0 = dist.Uniform(0, 1000).sample(torch.Size([nspots]))
    theta = torch.stack([x0, y0, sigma, N0, B0], dim=0)
    
    # Compute the likelihood of the data given the particle parameters
    loglike = mixloglike_torch(theta, adu, eta, texp, gain, var)
    #pyro.sample("obs", dist.Delta(loglike), obs=torch.tensor(0.0))

# Define the MCMC inference algorithm using NUTS
nuts_kernel = NUTS(model)

# Run the MCMC inference algorithm
num_samples = 100
num_chains = 1
mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_chains=num_chains)
mcmc_run = mcmc.run(adu, eta, texp, gain, var)
mcmc.summary()
print(mcmc_run.samples.items())

