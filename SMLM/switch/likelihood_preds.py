import numpy as np
from scipy.optimize import fminsearch

def likelihood_preds_m(Data, frames, m, model, FPR=None, nu=None):
    
    if np.linalg.norm(np.logical_not(np.logical_xor(Data, np.logical_not(Data.astype(bool)))), ord=1):
        print('Error: Data matrix must be binary.')
        return 
    
    if m not in [0, 1, 2]:
        print('Error: Can only estimate parameters for models with multiple off states m=0,1 or 2.')
        return 
    
    if len(model)-2 != m:
        print('Error: Incorrect specification of model with number of multiple off states m, must be of length m+2.')
        return 
    
    if np.linalg.norm(np.logical_not(np.logical_xor(model, np.logical_not(model.astype(bool)))), ord=1):
        print('Error: Incorrect specification of model, model must be a binary vector.')
        return 

    if FPR is None and nu is None:
        FPR = 1
        nu = 0
    elif FPR is not None and nu is None:
        FPR = bool(FPR)
        nu = 0
    elif FPR is not None and nu is not None:
        FPR = bool(FPR)
        if np.sum(nu) != 0:
            if len(nu)-3 != m:
                print('Error: Incorrect specification of initial probability mass, must be of length m+3.')
                return
            if np.sum(nu) != 1 or np.sum(nu < 0) > 0:
                print('Error: Incorrect specification of initial probability mass.')
                return
            if nu[-1] != 0:
                print('Error: Incorrect specification of initial probability mass, cannot start in the bleached state.')
                return
    elif FPR is None or nu is not None:
        raise ValueError('Error: invalid number of input parameters')

    other_params = np.concatenate((model.astype(int), [1, FPR]))
    Data = Data[np.sum(Data, axis=1) != 0, :]
    Delta = 1/frames

    if m == 0:
        lambda_start = initialisation(Data, Delta, model)
    else:
        lambdaold = initialisation(Data, Delta, [bool(np.sum(model[:-1])), bool(model[-1])])
        diff_scals = np.array([1/100, 1/80, 1/50, 1/40, 1/25, 1/5, 1/3, 1/2, *range(1, 11)])
        fval_exp = np.empty(len(diff_scals))
        eflag = np.empty(len(diff_scals))
        lambda_start = np.empty((len(diff_scals), 2*m+3))
        for sc in range(len(diff_scals)):
            lambda_start[sc, :], fval_exp[sc], eflag[sc] = ExampleFitDwellTimes_mstate(
                Data[np.sum(Data, axis=1) > 1, :], Delta, m, diff_scals[sc])
            lambda_start[sc, -2] = lambdaold[2]
        mu_start = np.concatenate((lambdaold[3] * model[:-1], lambdaold[4] * model[-1]), axis=None)
        lambda_start = lambda_start[np.isreal(fval_exp) & (eflag > 0), :]
        fval_exp = fval_exp[np.isreal(fval_exp) & (eflag > 0)]
    
	if not np.isclose(np.sum(nu), 0):
		no_nu_params_estimated = 0
		if m == 0:
		    zhat = z_trans_m([lambda_start[:2], lambda_start[2:4], Delta/100, 1e-7*FPR], Delta)
		else:
		    zhat = np.ones((len(fval_exp), 3*m+6))
		    fzs = np.arange(1, len(fval_exp)+1)
		    for sc in range(len(fval_exp)):
		        zhat[sc,:] = z_trans_m([lambda_start[sc,:-1], mu_start, Delta/100, 1e-7*FPR], Delta)
		        fzs[sc] = -eval_loglik_m(Data, zhat[sc,:], Delta, z_nu_trans_m(nu[:m+1]), m)
		    zhat = zhat[~np.isnan(fzs), :]
		    fzs = fzs[~np.isnan(fzs)]
		    J = np.argsort(-fzs)
		    zhat = zhat[J[-1], :]
		other_params = [model, 1, FPR]
		myfun = lambda z: -eval_loglik_m(Data, z[:2*(m+1)+len(other_params)], Delta, z_nu_trans_m(nu[:m+1]), m)
		options = {"maxiter": 20000, "maxfun": 20000}
		z, fval, eflag = fminsearch(myfun, zhat, options=options)
		z[np.isnan(z)] = -np.inf
		preds = lam_trans_m(z, Delta)
	else:
		no_nu_params_estimated = m+1
		nu_start = np.sum(Data[:,0] == 0) / len(Data[:,0])
		if np.isclose(nu_start, 0):
		    nu_start = 1e-6
		elif np.isclose(nu_start, 1):
		    nu_start = 1 - 1e-2
		if m == 0:
		    zhat = np.concatenate((z_trans_m([lambda_start[:2], lambda_start[2:4], Delta/100, 1e-7*FPR], Delta),
		                           z_nu_trans_m(nu_start)))
		else:
		    zhat = np.ones((len(fval_exp), 3*m+6))
		    fzs = np.arange(1, len(fval_exp)+1)
		    for sc in range(len(fval_exp)):
		        zhat[sc,:] = np.concatenate((z_trans_m([lambda_start[sc,:-1], mu_start, Delta/100, 1e-7*FPR], Delta),
		                                      z_nu_trans_m(np.concatenate((nu_start, 1e-6*np.ones(m)))))) 
		        fzs[sc] = -eval_loglik_m(Data, zhat[sc, :2*(m+1)+len(other_params)], Delta, zhat[sc, 2*(m+1)+len(other_params):3*(m+1)+len(other_params)], m)
		    zhat = zhat[~np.isnan(fzs), :]
		    fzs = fzs[~np.isnan(fzs)]
		    J = np.argsort(-fzs)
		    zhat = zhat[J[-1], :]
		other_params = [model, 1, FPR]
		myfun = lambda z: -eval_loglik_m(Data, z[:2*(m+1)+len(other_params)+1], Delta, z[2*(m+1)+len(other_params)+2:3*(m+1)+len(other_params)+1], m)
		# Negative log-likelihood function to be minimized.
		options = {'maxiter': 2e4, 'maxfev': 2e4}
		z, fval, eflag = fmin(myfun, zhat, options=options)
		z[np.isnan(z)] = -np.inf # Needed for parameters already known to the experimenter.
		preds = lam_trans_m(z[:2*(m+1)+len(other_params)+1], Delta) # Transforming back parameter estimates.
		nu_preds = nu_trans_m(z[2*(m+1)+len(other_params)+2:3*(m+1)+len(other_params)+1]) # Transforming back nu estimates.
		nu = np.concatenate((nu_preds, [1-sum(nu_preds), 0])) # Estimated nu vector.
		
		
	lmbda = preds[:2*(m+1)]
	mu = preds[2*(m+1):-2]
	delta = preds[-2]
	FPR = preds[-1]
	AIC = float('NaN')
	BIC = float('NaN')

	if eflag <= 0:
		print('Error: Not converged')
	else:
		no_params_estimated = no_nu_params_estimated + len(z[z!=-np.inf]) # Total number of estimated parameters.
		AIC = 2*no_params_estimated + 2*fval
		BIC = np.log(Data.size)*no_params_estimated + 2*fval


