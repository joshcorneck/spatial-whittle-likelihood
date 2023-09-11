#%%
import numpy as np
import pickle
import itertools

# from spatial_pp import SPP_Thomas

# Define grid and frequency parameters
minX = 0; minY = 0; maxX = 1; maxY = 1; ell = 1
area = (maxX - minX) * (maxY - minY)
freq_min = -8; freq_max = 8; freq_step = 1
freq_set = np.arange(freq_min, freq_max, freq_step)

# Sample a Thomas process
# rho = 50; K = 25; sigma = 0.03
# tom = SPP_Thomas()
# spps = []
# for i in range(1000):
#     spps.append(tom.simSPP(rho=rho, K=K, sigma=sigma, cov=np.array([[1,0], [0, 1]]), enlarge=1.25))

# with open('spps.pkl','wb') as f:
#      pickle.dump(spps, f)

with open('spps.pkl','rb') as f:
     spps = pickle.load(f)

spp = spps[0]
freq_min = -8; freq_max = 8; freq_step = 1
freq_set = np.arange(freq_min, freq_max, freq_step)
omega_set = np.array([x for x in itertools.product(freq_set, freq_set)])
#%%#
import time
import cython_periodogram 

start = time.time()
periodograms = []
for spp in spps[0:10]:
    taper_eval = cython_periodogram.orthog_sine_taper_py(spp[:,0],spp[:,1], 1, 1, 1)
    taper_pairwise_prod = cython_periodogram.pairwise_products_cpy(taper_eval)
    x_y_difference = cython_periodogram.pairwise_differences_cpy(spp)
    periodograms.append(cython_periodogram.compute_periodogram_cpy(omega_set.astype('float64'),
                                       freq_set.astype('float64'),
                                       x_y_difference,
                                       taper_pairwise_prod))

end = time.time()
print(f"Run time: {end - start}")
# %%
import numba_periodogram 

start = time.time()
periodograms = []
for spp in spps[0:5]:
    taper_eval = numba_periodogram.orthog_sine_taper_numba(spp[:,0],spp[:,1], 1, 1, 1)
    taper_pairwise_prod = numba_periodogram.pairwise_products_numba(taper_eval)
    x_y_difference = numba_periodogram.pairwise_differences_numba(spp)
    periodograms.append(numba_periodogram.compute_periodogram_numba(omega_set.astype('float64'),
                                       freq_set.astype('float64'),
                                       x_y_difference,
                                       taper_pairwise_prod))

end = time.time()
print(f"Run time: {end - start}")
# %%
