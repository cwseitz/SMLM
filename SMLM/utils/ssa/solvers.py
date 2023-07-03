import numpy as np
import matplotlib.pyplot as plt
from SSA._SSA import photoswitch
from .bin_ssa import bin_ssa, bin_lifetime
from scipy.integrate import odeint
from scipy.linalg import null_space, expm

class MasterSolver:
    def __init__(self,rates):
        self.num_states = 4
        self.rates = rates
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
    def current(self, time, P0):
        J = np.zeros((self.num_states, len(time)))
        for n, t in enumerate(time):
            dGtilde = self.generator @ expm(self.generator * t)
            J[:,n] = P0 @ dGtilde
        return J
    def solve(self, time, P0):
        P = np.zeros((self.num_states, len(time)))
        for n, t in enumerate(time):
            P[:,n] = P0.T @ expm(self.generator * t)
        return P
    def get_equilibrium(self):
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
            plt.plot(state[n,2,:])
            plt.show()
        print('Done.')
        return state  
        
    def solve(self,T,dt):
        X = self.ssa(T,dt)
        Xavg = np.mean(X,axis=0)
        return X, Xavg
        
    def plot(self,ax,P,t):
        ax.plot(t,P[0,:],color='pink',linestyle='--',label='1')
        ax.plot(t,P[1,:],color='cornflowerblue',linestyle='--',label='2')
        ax.plot(t,P[2,:],color='purple',linestyle='--',label='3')
        ax.plot(t,P[3,:],color='cyan',linestyle='--',label='4')
        ax.legend()
        
    def lifetime2s(self,X,dt):
        """Gets lifetimes for lumped 2-state system"""
        ns,nt = X.shape
        Xnew = np.zeros((2,nt))
        Xnew[0,:] = X[0,:]
        Xnew[1,:] = np.sum(X[1:,:],axis=0)
        X = Xnew
        times = []
        for i in range(2):
            times.append(np.array([]))
            counter = 0
            for j in range(nt):
                state = X[i,j]
                if state == 0:
                    times[i] = np.append(times[i],counter)
                    counter = 0
                else:
                    counter += dt
        for i in range(2):
            times[i] = times[i][times[i] > 0]
        return times

        
    def lifetime4s(self,X,dt):
        """Gets lifetimes for full 4-state system"""
        ns,nt = X.shape
        times = []
        for i in range(ns):
            times.append(np.array([]))
            counter = 0
            for j in range(nt):
                state = X[i,j]
                if state == 0:
                    times[i] = np.append(times[i],counter)
                    counter = 0
                else:
                    counter += dt
        for i in range(ns):
            times[i] = times[i][times[i] > 0]
        return times
