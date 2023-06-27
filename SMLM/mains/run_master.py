import numpy as np
import matplotlib.pyplot as plt
import time
from SMLM.utils import *

k12 = 0.1
k23 = 0.2
k34 = 0.12
k21 = 0.15
k31 = 0.04
k41 = 0.05

rates = np.array([k12,k23,k34,k21,k31,k41])
T = 100.0
dt = 0.001

###############
# Solution
###############

solver = MasterSolver(rates)
Pf = solver.solve()

###############
# SSA Solution
###############

time = np.arange(0,T,dt)
solver = SSASolver(rates)
P = solver.solve(T,dt)
solver.plot(P,time)
plt.show()


