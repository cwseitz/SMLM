from DNAFlex.psf import GaussianPSF, hessian_analytical, hessian_autograd, negloglike
from DNAFlex.stats.mcmc import MetropolisHastingsPSF
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt


def getCRLB(theta,counts,texp,eta,L=19,nchains=20,niter=1000,skip=5,tburn=500,take=500,plot=True):

    x0,y0,sigma,N0,B0 = theta
    mu = np.zeros((5,))
    var = np.array([0.01,0.01,0.01,10.0,10.0])
    cov = np.diag(var)
    collection = []
    for n in range(nchains):
        sampler = MetropolisHastingsPSF(negloglike,mu,cov,eta,texp)
        samples = sampler.run(counts,niter,theta,tburn=tburn,skip=skip)
        collection.append(samples)
        del sampler
    samples = np.concatenate(collection,axis=1)

    if plot:
        fig, ax = plt.subplots(1,5,figsize=(8,2))
        ax[0].hist(samples[0,:],color='black',density=True,bins=20)
        ax[0].set_xlabel(r'$x_{0}$')
        ax[1].hist(samples[1,:],color='black',bins=20,density=True)
        ax[1].set_xlabel(r'$y_{0}$')
        ax[2].hist(samples[2,:],color='black',bins=20,density=True)
        ax[2].set_xlabel(r'$\sigma$')
        ax[3].hist(samples[3,:],color='black',bins=20,density=True)
        ax[3].set_xlabel(r'$N_{0}$')
        ax[4].hist(samples[4,:],color='black',bins=20,density=True)
        ax[4].set_xlabel(r'$B_{0}$')
        ax[0].set_ylabel('Posterior')
        plt.tight_layout()
        plt.show()

    ntheta, nsamples = samples.shape
    samples = samples.T
    np.random.shuffle(samples)
    samples = samples.T
    samples = samples[:,:take]
    fisher = np.zeros((take,ntheta,ntheta))
    for n in range(take):
        fisher[n] = hessian_analytical(samples[:,n], counts, eta, texp)
    fisher_info = np.mean(fisher,axis=0)
    #CRLB = np.diagonal(np.linalg.inv(fisher_info))
    return samples, fisher

