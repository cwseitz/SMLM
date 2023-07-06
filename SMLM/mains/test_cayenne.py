from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
from cayenne.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
import time


###################
# Simulation
###################

tmin = 0.0; tmax = 1.0 #seconds
bin_size = 0.0001
bin_edges = np.arange(tmin,tmax,bin_size) #left edges
bin_centers = bin_edges + bin_size/2
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
        k23 = 0;
        k34 = 0;
        k21 = 80;
        k31 = 0;
        k41 = 0;
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

k12 = 450
k23 = 40.0
k34 = 5.0
k21 = 80
k31 = 4.0
k41 = 0.15

dd = np.array([])
nreps = 10000
for n in range(nreps):
    x1, x2, x3, x4, times = photoswitch([tmax,k12,k23,k34,k41,k31,k21])
    z = np.diff(x1*times)
    z = z[z > 0]
    dd = np.append(dd,z)

plt.hist(dd,bins=bin_edges,density=True)
plt.show()

