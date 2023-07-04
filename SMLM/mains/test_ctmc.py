from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
from cayenne.simulation import Simulation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import time

class CTMC:
    def __init__(self):
        pass
    def fit_onn_times(self,bins,vals):
        pass
    def fit_off_times(self,rinit,bins,times,plot=False):
        p, bins = np.histogram(times,bins=bins,density=False)
        bin_size = bin_edges[1]-bin_edges[0]       
        p = p.astype(np.float32)
        p = p/np.sum(p)    
        def loss(rates):
            G = np.zeros((4,4))
            G[0,1] = 0.0; G[1,2] = rates[0]
            G[2,3] = rates[1]; G[1,0] = rates[2]
            G[2,0] = rates[3]; G[3,0] = rates[4]
            for i in range(4):
                G[i,i] = -np.sum(G[i])
            J = np.zeros((4,len(bins)))
            P0 = np.array([0.0,1.0,0.0,0.0])   
            for n, t in enumerate(bins):
                dGtilde = G @ expm(G*t)
                J[:,n] = P0 @ dGtilde
            q = J[0,:]*bin_size
            L = np.sum((p-q[:-1])**2)
            self.plot(bin_edges,bin_size,p,q)
            return L

        bounds = [(0.01,500.0) for i in range(5)]
        result = minimize(loss,rinit,method='L-BFGS-B',bounds=bounds)
        if plot:
            self.plot(bin_edges,bin_size,p,q)
        return result
    def plot(self,bin_edges,bin_size,p,q):
        fig, ax = plt.subplots(1,2,figsize=(6,4))
        ax[0].bar(bin_edges[:-1], p, width=bin_size, align='edge')
        ax[1].bar(bin_edges, q, width=bin_size, align='edge')
        plt.show()

###################
# Params
###################

k12 = 0.0
k23 = 40.0
k34 = 5.0
k21 = 80.0
k31 = 4.0
k41 = 0.15

tmin = 0.0; tmax = 1.0 #seconds
bin_size = 0.001
bin_edges = np.arange(tmin,tmax,bin_size) #left edges

###################
# Simulation
###################

model_str = """
        const compartment comp1;
        comp1 = 1.0; # volume of compartment

        r1: A => B; k12;
        r2: B => C; k23;
        r3: C => D; k34;
        r4: B => A; k21;
        r5: C => A; k31;
        r6: D => A; k41;

        k12 = 450;
        k23 = 40;
        k34 = 5;
        k21 = 80;
        k31 = 4;
        k41 = 0.15;
        chem_flag = false;

        A = 1;
        B = 0;
        C = 0;
        D = 0;
    """

sim = Simulation.load_model(model_str, "ModelString")
sim.simulate(max_t=tmax, max_iter=1000, n_rep=10000)
results = sim.results
times = np.array([])
for x, t, status in results:
    state1 = x[:,0]
    z = state1*t
    z = z[z > 0]
    times = np.append(times,np.diff(z))

###################
# Fit
###################

#k23 = 20.0
#k34 = 1.0
#k21 = 50.0
#k31 = 2.0
#k41 = 0.5
rates_init = np.array([k23,k34,k21,k31,k41])
model = CTMC()
res = model.fit_off_times(rates_init,bin_edges,times)
print(res)
