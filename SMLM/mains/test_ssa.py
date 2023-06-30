from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
import matplotlib.pyplot as plt
import numpy as np

dt = 0.00001
T = 0.1 #seconds

k12 = 450 #s^-1
k23 = 40
k34 = 5
k21 = 80
k31 = 4
k41 = 0.15


rates = np.array([k12,k23,k34,k21,k31,k41])
solver = SSASolver(rates,nreps=100)
X,P = solver.solve(T,dt)
t = np.arange(0,T,dt)
fig,ax=plt.subplots()
solver.plot(ax,P,t)
ax.set_xlabel('Time (sec)')
plt.show()
