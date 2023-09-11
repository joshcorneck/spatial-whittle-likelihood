#%%
import numpy as np
import cmath
import matplotlib.pyplot as plt
import itertools
import time
import pickle

from numba import jit, njit, vectorize, float64

### NUMBA FUNCTIONS ###
    
# Compute orthogonal sine taper on the spp and store
@njit
def orthog_sine_taper_numba(x1, x2, p1, p2, ell):
    return (np.sin(np.pi * p1 * (x1 + ell/2)/ell) *
            np.sin(np.pi * p2 * (x2+ ell/2)/ell))

@njit
def pairwise_products_numba(vec: np.ndarray):
    k, size = 0, vec.size
    output = np.empty(size ** 2)
    for i in range(size):
        for j in range(size):
            output[k] = vec[i] * vec[j]
            k += 1
    return output

@njit
def pairwise_differences_numba(arr: np.ndarray):
    k, size = 0, len(arr)
    output = np.empty((size ** 2,2))
    for i in range(size):
        for j in range(size):
            output[k,:] = arr[i,:] - arr[j,:]
            k += 1
    return output

@vectorize([float64(float64)])
def vec_rect_numba(arg):
    return cmath.rect(1, arg).real

@njit
def compute_periodogram_numba(
    omega_set,
    freq_set,
    x_y_difference,
    taper_pairwise_prod
    ):
    len_freq_set = len(freq_set)
    I_p1p2 = np.zeros((len_freq_set, len_freq_set))
    k_w1, k_w2, count = -1, -1, -1
    for w in omega_set:
        count += 1
        k_w2 += 1
        if count // len_freq_set == count / len_freq_set:
            k_w1 += 1
            k_w2 = 0
        arg = (-2 * np.pi * w * x_y_difference).sum(axis=1)
        I_p1p2[k_w1, k_w2] = (taper_pairwise_prod * 
                            vec_rect_numba(arg)).sum()
    return I_p1p2

@njit
def average_periodogram_numba(spps, p1, p2, ell, freq_set, omega_set):
    periodograms = []
    ii = -1
    for spp in spps:
        ii +=1
        if (ii // 50 == ii / 50):
            print(f"Iteration: {ii + 1}")
        taper_eval = np.zeros(len(spp))
        taper_eval = (np.sin(np.pi * p1 * (spp[:,0] + ell/2)/ell) *
                        np.sin(np.pi * p2 * (spp[:,1] + ell/2)/ell))
        taper_pairwise_prod = pairwise_products_numba(taper_eval)
        x_y_difference = pairwise_differences_numba(spp)
        periodograms.append((compute_periodogram_numba(
            omega_set, 
            freq_set,
            x_y_difference,
            taper_pairwise_prod)))

#%%
