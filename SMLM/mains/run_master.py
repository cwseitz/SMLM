import numpy as np
import matplotlib.pyplot as plt
import time
from SMLM.utils import *
from SMLM.utils.ssa import *

k12 = 0.1
k23 = 0.2
k34 = 0.12
k21 = 0.15
k31 = 0.04
k41 = 0.05

rates = np.array([k12,k23,k34,k21,k31,k41])
T = 200.0
dt = 0.001

###############
# SSA Solution
###############

"""
time = np.arange(0,T,dt)
solver = SSASolver(rates)
P = solver.solve(T,dt)
Pl = np.zeros((2,int(round((T/dt)))))
Pl[0,:] = P[0,:]
Pl[1,:] = P[1,:]+P[2,:]+P[3,:]
solver.plot(P,time)
fig,ax=plt.subplots()
ax.plot(time,Pl[0,:])
ax.plot(time,Pl[1,:])
plt.show()
"""


###############
# Solution
###############

solver = MasterSolver(rates)
Pf = solver.solve()
Pf = np.array([0.25,0.25,0.25,0.25])

Pfa = get_pfa(*rates)
jac = rate_jac(*rates)

loss = cross_entropy_loss(rates,Pf)
print(loss)
