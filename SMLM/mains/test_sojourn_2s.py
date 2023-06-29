from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
import matplotlib.pyplot as plt
import numpy as np

#######################
# Params
#######################

dt = 0.00001
T = 0.1 #seconds

k12 = 450 #s^-1
k23 = 40
k34 = 5
k21 = 80
k31 = 4
k41 = 0.15


############################
# SSA solution
############################

nreps = 1000
rates = np.array([k12,k23,k34,k21,k31,k41])
solver = SSASolver(rates,nreps=nreps)
X,P = solver.solve(T,dt)

############################
# Sojourn time distributions
############################

x1times = np.array([])
x2times = np.array([])

for n in range(nreps):
    Xn = X[n,:]
    times = solver.lifetime2s(Xn,dt)
    x1times = np.concatenate([x1times,times[0]])
    x2times = np.concatenate([x2times,times[1]])

############################
# Analytical OFF time 
############################

t = np.arange(0,0.1,dt)
lam1 = k21 + k23; lam2 = k31 + k34; lam3 = k41
A = lam2-lam3; B = lam3-lam1; C = lam1 - lam2
D = (lam2-lam3)*lam2*lam3 + (lam3-lam1)*lam1*lam3 + (lam1-lam2)*lam1*lam2
a1 = 1 + k23/(lam2-lam1) + (k23*k23*A)/D
a2 = -k23/(lam2-lam1) + (k23*k23*B)/D
a3 = (k23*k23*C)/D
foff = a1*lam1*np.exp(-lam1*t) + a2*lam2*np.exp(-lam2*t) + a3*lam3*np.exp(-lam3*t)


############################
# Plots
############################

bins = np.linspace(0,0.1,20)
fig,ax=plt.subplots(1,2,figsize=(10,3))
vals1, bins1 = np.histogram(x1times,bins=bins,density=True)
vals2, bins2 = np.histogram(x2times,bins=bins,density=True)
ax[0].plot(bins1[:-1],vals1,color='cornflowerblue',label='SSA')
ax[1].plot(bins2[:-1],vals2,color='cornflowerblue',label='SSA')
ax[1].plot(t,foff,color='red',label='Theory')
ax[0].set_xlabel('ON lifetime (sec)')
ax[1].set_xlabel('OFF lifetime (sec)')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

