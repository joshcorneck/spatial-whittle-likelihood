#%%
import numpy as np
import cmath
import matplotlib.pyplot as plt
import itertools
import time
import pickle

import cython
cimport numpy as np


### CYTHON FUNCTIONS ###

cpdef pairwise_differences_cpy(np.ndarray[np.float64_t, ndim=2] arr):
    cdef Py_ssize_t size = arr.shape[0]
    cdef np.ndarray output = np.empty((size * (size - 1) / 2 + size, 2), dtype=np.float64)

    cdef int i, j, k, m
    m = 0
    for i in range(size):
        for j in range(i + 1):
            for k in range(1 + 1):
                output[m,k] = arr[i,k] - arr[j,k]
            m += 1
    return output   

def orthog_sine_taper_py(x1, x2, p1, p2, ell):
     return (np.sin(np.pi * p1 * (x1 + ell/2)/ell) *
             np.sin(np.pi * p2 * (x2+ ell/2)/ell))

cpdef pairwise_products_cpy(np.ndarray[np.float64_t, ndim=1] vec):
    cdef int size = vec.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] output = np.empty(size * (size - 1) / 2 + size)
    
    cdef int i, j, k
    k = 0
    for i in range(size):
        for j in range(i + 1):
            output[k] = vec[i] * vec[j]
            k += 1
    return output

cpdef compute_periodogram_cpy(
    np.ndarray[np.float64_t, ndim=2] omega_set,
    np.ndarray[np.float64_t, ndim=1] freq_set,
    np.ndarray[np.float64_t, ndim=2] x_y_difference,
    np.ndarray[np.float64_t, ndim=1] taper_pairwise_prod
    ):
    cdef int len_freq_set = len(freq_set)
    cdef np.ndarray[np.float64_t, ndim=2] I_p1p2 = np.zeros((len_freq_set, len_freq_set))
    cdef int k_w1, k_w2, count 
    k_w1, k_w2, count = -1, -1, -1
    cdef np.ndarray[np.float64_t, ndim=1] w
    cdef np.ndarray[np.float64_t, ndim=1] cos_arg
    for w in omega_set:
        count += 1
        k_w2 += 1
        if count % len_freq_set == 0:
            k_w1 += 1
            k_w2 = 0
        cos_arg = np.cos((2 * np.pi * w * x_y_difference).sum(axis=1))
        I_p1p2[k_w1, k_w2] = 2 * (taper_pairwise_prod * cos_arg).sum()
    return I_p1p2
