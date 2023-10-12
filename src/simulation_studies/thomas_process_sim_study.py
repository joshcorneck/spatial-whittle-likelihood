# My module imports
from simulation_methods.spatial_pp import SPP_Thomas
from periodogram_methods.periodogram import MultiOrthogSineTaper, DebiasedBartlettPeriodogram
from fitting_methods.whittle_estimator import ThomasWhittleEstimator

# External module imports
import numpy as np
import warnings
import argparse
import pickle

"""
This simulation study works as follows:

    We run the Whittle likelihood inference procedure on a Thomas 
    process with n = 25, 50, 100, 200, 400, 800 points, with a 
    constant lambda = 0.01. In this way, we select 
    ell_n = sqrt(n/0.01). We will run this for two sets of parameter
    values:
        1. rho = 0.6lambda; K = lambda/rho; sigma = 2
        2. rho = 0.3lambda; K = lambda/rho; sigma = 6
    These correspond to tight and spread Thomas processes, respectively.
    We will run the inference N = 1000 times, simulating from a Thomas
    process with those parameter values each time.
"""

##### PARSE ARGUMENTS #####
parser = argparse.ArgumentParser()
parser.add_argument("N", type=int, help="number of iterations")
parser.add_argument("n", type=int, help="average number of points")
parser.add_argument("param_set_choice", type=int, help="parameter choice (0: sig=2, 1: sig=6)")
parser.add_argument("periodogram_choice", type=int, help="periodogram choice (0: Bart, 1: sine)")
args = parser.parse_args()

##### SIMULATION PARAMETERS ######
N = args.N; n = args.n; param_set_choice = args.param_set_choice; per_choice = args.periodogram_choice
freq_min = -0.2; freq_max = 0.2; freq_step = 0.005

##### GLOBAL THOMAS PROCESS PARAMETERS ######
lambdaHom = 0.01

##### PARAMETER SETS ######
param_set = np.array([[0.6*lambdaHom, 2],
                      [0.3*lambdaHom, 6]])
rho = param_set[param_set_choice, 0]; sigma = param_set[param_set_choice, 1]

###### ARRAYS/LISTS TO HOLD ESTIMATES/SIMULATIONS ######
param_estimates = np.zeros((N, 3))
param_init = np.zeros((N,2))
spps = []

###### LOCAL THOMAS PROCESS PARAMETERS #######
elln = (n / lambdaHom) ** (0.5); K = lambdaHom / rho

print(f"... Parameter values: rho = {rho}, K = {K}, sigma = {sigma} ...")

###### INSTANTIATE CLASSES ######
thom = SPP_Thomas(minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2)
if per_choice == 0:
    per = DebiasedBartlettPeriodogram(freq_min=freq_min, freq_max=freq_max, freq_step=freq_step,
                            minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2)
else:
    per = MultiOrthogSineTaper(freq_min=freq_min, freq_max=freq_max, freq_step=freq_step,
                            minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2)


###### RUN N SIMULATIONS ######
for i, run in enumerate(range(N)):
    
    print(f"Run: {run + 1}")
    
    ##### SIMULATE THOMAS PROCESS AND COMPUTE PERIODOGRAM #####
    thom_spp = thom.simSPP(rho=rho, K=K, sigma=sigma, 
                            cov=np.array([[1, 0], [0, 1]]), 
                            enlarge=1.25)
    spps.append(thom_spp)

    if per_choice == 0:
        per.computeSinglePeriodogram(thom_spp)
    else:
        per.computeMultitaperSinglePeriodogram(thom_spp, P=3)
    periodogram = per.periodogram

    ##### INSTANTIATE WHITTLE ESTIMATOR AND NEGATIVE LL FUNCTION ####
    lam_hat = len(thom_spp) / (elln ** 2)
    twe = ThomasWhittleEstimator(spp=thom_spp, periodogram=periodogram,
                                freq_min=freq_min, freq_max=freq_max, freq_step=freq_step, 
                                minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2 
                                )
    neg_ll_orthog = lambda x: (
        -twe.computeLikelihood(lam_hat=lam_hat, rho_tilde=x[0], sig_tilde=x[1])
    ) 
    neg_jac = lambda x: (
        -twe.computeLikelihoodDerivative(lam_hat=lam_hat, rho_tilde=x[0], sig_tilde=x[1])
    )                            

    ##### RANDOMLY INITIALISE AND COMPUTE THE MINIMUM #####
    rho_init = np.random.uniform(-6, -3, 1)
    sig_init = np.random.uniform(0.5, 1.5, 1)
    x_tilde_init = np.array([rho_init, sig_init]).reshape((2,))

    # Use a context manager to temporarily suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        
        soln_orthog = twe.scipyOptimisation(neg_ll_orthog, x_tilde_init,
                                            method='BFGS', jac=neg_jac, options={'disp': False})
    
    ###### STORE PARAMETERS #######
    print(f"rho: {np.exp(soln_orthog[0])}, K: {lam_hat / np.exp(soln_orthog[0])}, sigma: {np.exp(soln_orthog[1])}")
    rho_est = np.exp(soln_orthog[0]); K_est = lam_hat / np.exp(soln_orthog[0])
    sig_est = np.exp(soln_orthog[1])
    param_estimates[i,:] = np.array([rho_est, K_est, sig_est])
    param_init[i,:] = x_tilde_init


##### SAVE DOWN FILES ######
with open('param_estimates' + f'_n{n}' + f'_per{per_choice}' + f'_set{param_set_choice}' + '.pkl','wb') as f:
    pickle.dump(param_estimates, f)
    print("param_estimates saved.")
f.close()

with open('param_inits' + f'_n{n}' + f'_per{per_choice}' + f'_set{param_set_choice}' + '.pkl','wb') as f:    
    pickle.dump(param_init, f)
    print("param_init saved.")
f.close()

with open('spps' + f'_n{n}' + f'_per{per_choice}' + f'_set{param_set_choice}' + '.pkl','wb') as f: 
    pickle.dump(spps, f)
    print("spps saved.")
f.close()
