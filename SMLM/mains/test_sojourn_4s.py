from SSA._SSA import photoswitch
from SMLM.utils.ssa import *
import matplotlib.pyplot as plt
import numpy as np

dt = 0.00001
T = 0.5 #seconds

k12 = 450 #s^-1
k23 = 40
k34 = 5
k21 = 80
k31 = 4
k41 = 0.15

nreps = 1000
rates = np.array([k12,k23,k34,k21,k31,k41])
solver = SSASolver(rates,nreps=nreps)
X,P = solver.solve(T,dt)

#times = solver.lifetime4s(X[0],dt)

############################
# Sojourn time distributions
############################


x1times = np.array([])
x2times = np.array([])
x3times = np.array([])
x4times = np.array([])

for n in range(nreps):
    Xn = X[n,:]
    times = solver.lifetime4s(Xn,dt)
    #print(times)
    #plt.plot(Xn[0])
    #plt.plot(Xn[1])
    #plt.plot(Xn[2])
    #plt.plot(Xn[3])
    #plt.show()
    x1times = np.concatenate([x1times,times[0]])
    x2times = np.concatenate([x2times,times[1]])
    x3times = np.concatenate([x3times,times[2]])
    x4times = np.concatenate([x4times,times[3]])


############################
# Plots
############################

t = np.linspace(0,0.05,1000) #seconds
exp1 = np.exp(-k12 * t)
exp2 = np.exp(-(k23 + k21) * t)
exp3 = np.exp(-(k34 * k31) * t)
exp4 = np.exp(-k41 * t)


fig,ax=plt.subplots(1,4,figsize=(10,3))

vals1, bins1 = np.histogram(x1times,bins=20,density=True)
vals2, bins2 = np.histogram(x2times,bins=20,density=True)
vals3, bins3 = np.histogram(x3times,bins=20,density=True)
vals4, bins4 = np.histogram(x4times,bins=20,density=True)

ax[0].plot(bins1[:-1],vals1,color='cornflowerblue',label='SSA')
ax[1].plot(bins2[:-1],vals2,color='cornflowerblue',label='SSA')
ax[2].plot(bins3[:-1],vals3,color='cornflowerblue',label='SSA')
ax[3].plot(bins4[:-1],vals4,color='cornflowerblue',label='SSA')


ax[0].plot(t, exp1*vals1.max(), color='red', label='Theory')
ax[1].plot(t, exp2*vals2.max(), color='red', label='Theory')
ax[2].plot(t, exp3*vals3.max(), color='red', label='Theory')
ax[3].plot(t, exp4*vals4.max(), color='red', label='Theory')


ax[0].set_xlabel('State 1 lifetime (sec)')
ax[1].set_xlabel('State 2 lifetime (sec)')
ax[2].set_xlabel('State 3 lifetime (sec)')
ax[3].set_xlabel('State 4 lifetime (sec)')

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
plt.tight_layout()
plt.show()

