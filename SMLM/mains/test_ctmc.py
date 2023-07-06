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
    def forward(self,rates,qbins):
        G = np.zeros((4,4))
        G[0,1] = 0.0; G[1,2] = rates[0]
        G[2,3] = rates[1]; G[1,0] = rates[2]
        G[2,0] = rates[3]; G[3,0] = rates[4]
        for i in range(4):
            G[i,i] = -np.sum(G[i])
        J = np.zeros((4,len(qbins)))
        P0 = np.array([0.0,1.0,0.0,0.0])   
        for n, t in enumerate(qbins):
            dGtilde = G @ expm(G*t)
            J[:,n] = P0 @ dGtilde
        q = J[0,:]*q_bin_size  
        return q      
    def loss(self,p,rates,pbins,qbins):
        q = self.forward(rates,qbins)
        qreduced = np.sum(q.reshape(100,100),axis=1)
        self.plot(pbins,pbins,p,qreduced)
        return np.sum((p[2:]-qreduced[2:-1])**2)
    def fit_off_times(self,p,rinit,pbins,qbins,times,plot=False):
        q = self.forward(rinit,qbins)
        qreduced = np.sum(q.reshape(100,100),axis=1)
        self.plot(pbins,pbins,p,qreduced)
    def plot(self,pbins,qbins,p,q):
        p_bin_size = pbins[1]-pbins[0] 
        q_bin_size = qbins[1]-qbins[0]
        plt.plot(p-q[:-1])
        fig, ax = plt.subplots(1,2,figsize=(6,4),sharex=True,sharey=True)
        ax[0].bar(pbins[:-1], p, width=p_bin_size, align='edge')
        ax[1].bar(qbins, q, width=q_bin_size, align='edge')
        plt.show()


###################
# Simulation
###################

tmin = 0.0; tmax = 1.0 #seconds

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
        k23 = 40.0;
        k34 = 5.0;
        k21 = 80;
        k31 = 4.0;
        k41 = 0.15;
        chem_flag = false;

        A = 1;
        B = 0;
        C = 0;
        D = 0;
    """

sim = Simulation.load_model(model_str, "ModelString")
sim.simulate(max_t=tmax, max_iter=10000, n_rep=10000, algorithm="direct")
results = sim.results
times = np.array([])
for x, t, status in results:
    state1 = x[:,0]
    z = state1*t
    z = z[z > 0]
    d = np.diff(z)
    times = np.append(times,d)


###################
# Fit
###################


k12 = 450
k23 = 40.0
k34 = 5.0
k21 = 80
k31 = 4.0
k41 = 0.15


k12init = 0.0
k23init = 20.0
k34init = 1.0
k21init = 60.0
k31init = 2.0
k41init = 0.1

p_bin_size = 0.01
p_bins = np.arange(tmin,tmax,p_bin_size)
q_bin_size = 0.0001
q_bins = np.arange(tmin,tmax,q_bin_size) + q_bin_size/2
rates = np.array([k23,k34,k21,k31,k41])
rates_init = np.array([k23init,k34init,k21init,k31init,k41init])
p, bins = np.histogram(times,bins=p_bins,density=False)      
p = p.astype(np.float32)
p = p/np.sum(p)

model = CTMC()
print(model.loss(p,rates_init,p_bins,q_bins))
print(model.loss(p,rates,p_bins,q_bins))
#res = model.fit_off_times(rates_init,p_bins,q_bins,times,plot=True)

