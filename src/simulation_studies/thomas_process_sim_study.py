# My module imports
from simulation_methods.spatial_pp import SPP_Thomas
from periodogram_methods.periodogram import MultiOrthogSineTaper
from fitting_methods.whittle_estimator import ThomasWhittleEstimator
import pickle

# External module imports
import numpy as np
import warnings

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

##### SIMULATION PARAMETERS ######
N = 1; n = 25
freq_min = -0.25; freq_max = 0.25; freq_step = 0.005

##### GLOBAL THOMAS PROCESS PARAMETERS ######
lambdaHom = 0.01

##### PARAMETER SETS ######
param_set = np.array([[0.6*lambdaHom, 2],
                      [0.3*lambdaHom, 6]])


for rho, sigma in param_set:

    print(f"... Parameter values: rho = {rho}, sigma = {sigma} ...")

    ###### LOCAL THOMAS PROCESS PARAMETERS #######
    elln = (n / lambdaHom) ** (0.5); K = lambdaHom / rho

    ###### INSTANTIATE CLASSES ######
    thom = SPP_Thomas(minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2)
    per = MultiOrthogSineTaper(freq_min=freq_min, freq_max=freq_max, freq_step=freq_step,
                               minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2)

    ###### ARRAYS/LISTS TO HOLD ESTIMATES/SIMULATIONS ######
    param_estimates = np.zeros((N, 3))
    param_init = np.zeros((N,3))
    spps = []

    for i, run in enumerate(range(N)):
        
        print(f"Run: {run + 1}")
        
        ##### SIMULATE THOMAS PROCESS AND COMPUTE PERIODOGRAM #####
        thom_spp = thom.simSPP(rho=rho, K=K, sigma=sigma, 
                               cov=np.array([[1, 0], [0, 1]]), 
                               enlarge=1.25)
        spps.append(thom_spp)
        per.computeMultitaperSinglePeriodogram(thom_spp, P=3)
        orthog_sine_periodogram = per.periodogram

        ##### INSTANTIATE WHITTLE ESTIMATOR AND NEGATIVE LL FUNCTION ####
        twe = ThomasWhittleEstimator(spp=thom_spp, 
                                    freq_min=freq_min, freq_max=freq_max, freq_step=freq_step, 
                                    minX=-elln/2, maxX=elln/2, minY = -elln/2, maxY = elln/2 
                                    )
        neg_ll_orthog = lambda x: (
            -twe.computeLikelihood(periodogram=orthog_sine_periodogram,
                                   rho=x[0], K=x[1], sigma=x[2])
                                  )

        ##### RANDOMLY INITIALISE AND COMPUTE THE MINIMUM #####
        rho_init = np.random.uniform(0.001, 0.3, 1)
        K_init = np.random.uniform(0.5, 3.5, 1)
        sig_init = np.random.uniform(1, 5, 1)
        x_init = np.array([rho_init, K_init, sig_init]).reshape((3,))

        # Use a context manager to temporarily suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            soln_orthog = twe.scipyOptimisation(neg_ll_orthog, x_init)
        
        ###### STORE PARAMETERS #######
        param_estimates[i,:] = soln_orthog
        param_init[i,:] = x_init


##### SAVE DOWN FILES ######
with open('./simulation_studies/param_estimates' + f'_n{n}' + '.pkl','wb') as f:
    pickle.dump(param_estimates, f)
    print("param_estimates saved.")
f.close()

with open('./simulation_studies/param_inits' + f'_n{n}' + '.pkl','wb') as f:    
    pickle.dump(param_init, f)
    print("param_init saved.")
f.close()

with open('./simulation_studies/spps' + f'_n{n}' + '.pkl','wb') as f: 
    pickle.dump(spps, f)
    print("spps saved.")
f.close()
