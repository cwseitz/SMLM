import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from SSA._SSA import photoswitch
from .ssa import bin_ssa, lifetime4s, bin_lifetime
from scipy.integrate import odeint
from scipy.linalg import null_space


class MasterSolver:
    def __init__(self,rates):
        self.num_states = 4
        self.rates = rates
    def solve(self):
        transition_matrix = np.zeros((self.num_states,self.num_states))
        generator_matrix = np.zeros((self.num_states,self.num_states))
        transition_matrix[0,1] = self.rates[0]
        transition_matrix[1,2] = self.rates[1]
        transition_matrix[2,3] = self.rates[2]
        transition_matrix[1,0] = self.rates[3]
        transition_matrix[2,0] = self.rates[4]
        transition_matrix[3,0] = self.rates[5]
        generator_matrix += transition_matrix
        for i in range(self.num_states):
            generator_matrix[i,i] = -np.sum(generator_matrix[i])
        self.generator = generator_matrix
        stationary_matrix = null_space(self.generator.T)
        stationary_vector = stationary_matrix[:, 0]
        stationary_vector /= np.sum(stationary_vector)
        return stationary_vector
    
class SSASolver:
    def __init__(self,rates,nreps=1000):
        self.rates = rates
        self.nreps = nreps
    def ssa(self,T,dt):
        print('Simulating photoswitching with SSA...')
        k12,k23,k34,k21,k31,k41 = self.rates
        nt = int(round(T/dt))
        state = np.zeros((self.nreps,4,nt),dtype=np.bool)
        for n in range(self.nreps):
            x1, x2, x3, x4, times = photoswitch([T,k12,k23,k34,k41,k31,k21])
            t_bins, x1_binned, x2_binned, x3_binned, x4_binned = bin_ssa(times,x1,x2,x3,x4,dt,T)
            state[n,0,:] = x1_binned
            state[n,1,:] = x2_binned
            state[n,2,:] = x3_binned
            state[n,3,:] = x4_binned
        print('Done.')
        return state  
    def solve(self,T,dt):
        X = self.ssa(T,dt)
        times1, times2 = lifetime4s(X,dt)
        bins = np.arange(0,100,1)
        vals0, vals1 = bin_lifetime(times1,times2,bins,density=True)
        Xavg = np.mean(X,axis=0)
        return Xavg
    def plot(self,P,t):
        fig,ax=plt.subplots()
        ax.plot(t,P[0,:],color='pink',linestyle='--',label='1')
        ax.plot(t,P[1,:],color='cornflowerblue',linestyle='--',label='2')
        ax.plot(t,P[2,:],color='purple',linestyle='--',label='3')
        ax.plot(t,P[3,:],color='cyan',linestyle='--',label='4')
        ax.legend()
