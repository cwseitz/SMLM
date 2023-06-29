import numpy as np
import matplotlib.pyplot as plt
import time
from SMLM.utils import *
from SMLM.utils.ssa import *
from scipy.optimize import minimize

###############
# Ground truth
###############

k12 = 450
k23 = 30
k34 = 5
k21 = 80
k31 = 4
k41 = 0.1

rates_true = np.array([k12,k23,k34,k21,k31,k41])
p_obs = get_pfa(*rates_true)
print(-1*np.sum(p_obs*np.log(p_obs)))

###############
# Optimizer
###############

k12 = 10.0
k23 = 10.0
k34 = 10.0
k21 = 10.0
k31 = 10.0
k41 = 10.0

rates_initial = np.array([k12,k23,k34,k21,k31,k41])

def objective_func(rates):
    return cross_entropy_loss(rates, p_obs)
def gradient_func(rates):
    q = get_pfa(*rates)
    return cross_entropy_grad(rates, p_obs, q)

bounds = [(0.01,500.0) for i in range(6)]
result = minimize(objective_func, rates_initial, method='L-BFGS-B', jac=gradient_func, bounds=bounds)
rates_opt = result['x']
print(rates_opt)
q = get_pfa(*rates_opt)
print(p_obs,q)

#################################
# Plot opt sojourn v true sojourn
#################################

t = np.linspace(0,0.05,1000) #seconds

exp1true = np.exp(-rates_true[0] * t)
exp2true = np.exp(-(rates_true[1] + rates_true[3]) * t)
exp3true = np.exp(-(rates_true[2] * rates_true[4]) * t)
exp4true = np.exp(-rates_true[-1] * t)

exp1opt = np.exp(-rates_opt[0] * t)
exp2opt = np.exp(-(rates_opt[1] + rates_opt[3]) * t)
exp3opt = np.exp(-(rates_opt[2] * rates_opt[4]) * t)
exp4opt = np.exp(-rates_opt[-1] * t)

fig,ax=plt.subplots(1,4)

ax[0].plot(t, exp1true, color='red', label='True')
ax[1].plot(t, exp2true, color='red', label='True')
ax[2].plot(t, exp3true, color='red', label='True')
ax[3].plot(t, exp4true, color='red', label='True')
ax[0].plot(t, exp1opt, color='blue', label='Opt')
ax[1].plot(t, exp2opt, color='blue', label='Opt')
ax[2].plot(t, exp3opt, color='blue', label='Opt')
ax[3].plot(t, exp4opt, color='blue', label='Opt')
plt.show()
