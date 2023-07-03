from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
from cayenne.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np

k12 = 0.0
k23 = 40.0
k34 = 5.0
k21 = 80.0
k31 = 4.0
k41 = 0.15

##########################
# Compute J at bin centers
##########################

rates = np.array([k12,k23,k34,k21,k31,k41])
solver = MasterSolver(rates)
tmin = 0.0
tmax = 1.0 #seconds
bin_size = 0.001
bin_edges = np.arange(tmin,tmax,bin_size) #left edges
bin_centers = bin_edges + bin_size/2
P0 = np.array([0.0,1.0,0.0,0.0])

###########################################
# Estimate histogram using J at bin centers
###########################################

J = solver.current(bin_edges,P0)
h = J[0,:]*bin_size
print(np.sum(h))
fig, ax = plt.subplots(1,2,figsize=(6,4),sharex=True,sharey=True)
ax[0].bar(bin_edges, h, width=bin_size, align='edge')
ax[0].set_xlabel('Bins')

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

vals, bins = np.histogram(times,bins=bin_edges,density=True)
vals = vals.astype(np.float32)
vals = vals/np.sum(vals)
ax[1].bar(bin_edges[:-1], vals, width=bin_size, align='edge')
ax[1].set_xlabel('Bins')

plt.tight_layout()
plt.show()
