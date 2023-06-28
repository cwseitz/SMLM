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


