from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
import matplotlib.pyplot as plt
import numpy as np

dt = 0.00001
T = 0.1 #seconds

#k12 = 450 #s^-1
k12 = 0.0
k23 = 40.0
k34 = 5.0
k21 = 80.0
k31 = 4.0
k41 = 0.15


rates = np.array([k12,k23,k34,k21,k31,k41])
solver = MasterSolver(rates)
t = np.linspace(0,10.0,1000)
P0 = np.array([0.0,1.0,0.0,0.0])
P = solver.solve(t,P0)

fig,ax=plt.subplots()
ax.plot(t,1-P[0])
#ax.plot(P[1])
#ax.plot(P[2])
#ax.plot(P[3])
ax.set_ylabel('OFF fraction')
ax.set_xlabel('Time (sec)')
plt.show()
