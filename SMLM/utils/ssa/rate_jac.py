from .rate_grad import *

"""Null space of generator matrix"""

def get_pfa(k12, k23, k34, k21, k31, k41):
    Pfa = np.zeros((4,))
    Pfa[0] = (k21*k31*k41 + k21*k34*k41 + k23*k31*k41 + k23*k34*k41)/(k12*k23*k34)
    Pfa[1] = (k31*k41 + k34*k41)/(k23*k34)
    Pfa[2] = k41/k34
    Pfa[3] = 1
    Pfa /= np.sum(Pfa)
    return Pfa

def cross_entropy_loss(rates, p_obs):
    q = get_pfa(*rates)
    return -1*np.sum(p_obs*np.log(q))


def cross_entropy_grad(rates,p,q):
    ratio = p/q
    J = rate_jac(*rates)
    return -1*np.sum(J*ratio[:,np.newaxis],axis=0)

def rate_jac(k12, k23, k34, k21, k31, k41):

    jac = np.zeros((4,6))
    jac[0,0] = gradient_Pfa0_k12(k12, k23, k34, k21, k31, k41)
    jac[0,1] = gradient_Pfa0_k23(k12, k23, k34, k21, k31, k41)
    jac[0,2] = gradient_Pfa0_k34(k12, k23, k34, k21, k31, k41)
    jac[0,3] = gradient_Pfa0_k21(k12, k23, k34, k21, k31, k41)
    jac[0,4] = gradient_Pfa0_k31(k12, k23, k34, k21, k31, k41)
    jac[0,5] = gradient_Pfa0_k41(k12, k23, k34, k21, k31, k41)
    
    jac[1,0] = gradient_Pfa1_k12(k12, k23, k34, k21, k31, k41)
    jac[1,1] = gradient_Pfa1_k23(k12, k23, k34, k21, k31, k41)
    jac[1,2] = gradient_Pfa1_k34(k12, k23, k34, k21, k31, k41)
    jac[1,3] = gradient_Pfa1_k21(k12, k23, k34, k21, k31, k41)
    jac[1,4] = gradient_Pfa1_k31(k12, k23, k34, k21, k31, k41)
    jac[1,5] = gradient_Pfa1_k41(k12, k23, k34, k21, k31, k41)
    
    jac[2,0] = gradient_Pfa2_k12(k12, k23, k34, k21, k31, k41)
    jac[2,1] = gradient_Pfa2_k23(k12, k23, k34, k21, k31, k41)
    jac[2,2] = gradient_Pfa2_k34(k12, k23, k34, k21, k31, k41)
    jac[2,3] = gradient_Pfa2_k21(k12, k23, k34, k21, k31, k41)
    jac[2,4] = gradient_Pfa2_k31(k12, k23, k34, k21, k31, k41)
    jac[2,5] = gradient_Pfa2_k41(k12, k23, k34, k21, k31, k41)
    
    jac[3,0] = gradient_Pfa3_k12(k12, k23, k34, k21, k31, k41)
    jac[3,1] = gradient_Pfa3_k23(k12, k23, k34, k21, k31, k41)
    jac[3,2] = gradient_Pfa3_k34(k12, k23, k34, k21, k31, k41)
    jac[3,3] = gradient_Pfa3_k21(k12, k23, k34, k21, k31, k41)
    jac[3,4] = gradient_Pfa3_k31(k12, k23, k34, k21, k31, k41)
    jac[3,5] = gradient_Pfa3_k41(k12, k23, k34, k21, k31, k41)
    
    return jac
