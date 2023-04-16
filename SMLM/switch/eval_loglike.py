import numpy as np
from emission_matrices import (
    emission_delta_endstate_3state,
    emission_delta_endstate_4state,
    emission_delta_endstate_5state,
)
from forward_backward import fwdbkwd_norm_c2
from transition_matrices import lam_trans_m, nu_trans_m


def eval_loglik_m(X, lambdas, Delta, nu, m):
    """
    This function evaluates the log-likelihood function for given lambdas,
    mus, and other nuisance parameters. Here lambda is a vector with all
    transition rates between states, all absorption rates, delta and FPR.

    Args:
        X (ndarray): An array of size (s[0], n), where s[0] is the number of
            independent emitters and n is the number of observations for each
            emitter.
        lambdas (ndarray): A vector with all transition rates between states,
            all absorption rates, delta, and FPR.
        Delta (float): A scaling factor for the rates. Must be greater than
            zero.
        nu (ndarray): A vector of nu parameters.
        m (int): An integer indicating the number of hidden states.

    Returns:
        float: The log-likelihood.
    """
    s = X.shape
    likelihood = np.ones(s[0])

    # Transform lambda and nu back from real line.
    lambdas = lam_trans_m(lambdas, Delta)
    lambdas[np.isnan(lambdas)] = 0
    nu = nu_trans_m(nu)
    nu[np.isnan(nu)] = 0
    nu = np.concatenate((nu, [1 - sum(nu), 0]))

    # Extract lambda, mu, delta, and FPR from lambdas.
    lambda_ = lambdas[: 2 * (m + 1)]
    mu = lambdas[2 * (m + 1) : -2]
    delta = lambdas[-2]
    FPR = lambdas[-1]

    # Get emission matrices needed in the log-likelihood from the lambda, mu,
    # nu vectors, and delta (noise parameter).
    if m == 0:
        BP0, BP1 = emission_delta_endstate_3state(lambda_, mu, delta, Delta)
    elif m == 1:
        BP0, BP1 = emission_delta_endstate_4state(lambda_, mu, delta, Delta)
    elif m == 2:
        BP0, BP1 = emission_delta_endstate_5state(lambda_, mu, delta, Delta)

    # Update emission matrices to allow for the false positive rate (FPR) which
    # is given by lambdas(end).
    BP0_FPR = (1 - FPR) * BP0  # emission matrix when a 0 is seen.
    BP1_FPR = BP1 + FPR * BP0  # emission matrix when a 1 is seen.

    # Calculate likelihood for each independent emitter.
    for i in range(s[0]):
        likelihood[i] = fwdbkwd_norm_c2(X[i, :], nu, BP0_FPR, BP1_FPR)

    # Sum up the likelihood for all the independent emitters.
    likelihood = np.sum(likelihood)

    # Return log-likelihood.
    return likelihood

