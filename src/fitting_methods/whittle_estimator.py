#%%
import numpy as np
import scipy, time
from scipy.optimize import minimize
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Union
import mpmath as mp
import math

from spatial_pp import SPP_Thomas
from periodogram import Periodogram

class BaseWhittleEstimator(Periodogram, ABC):
    """
    A base class to compute an estimator based on the Whittle likelihood.
    Parameters:
        - spp: the spatial point process (np.array).
        - freq_min, freq_max, freq_step: integers defining range and 
                step size of the frequencies to evaluate at.
        - minX, maxX, ...: grid we simulate on.
        - taper: normalised taper function. If not supplied, the default is to 
             leave as None and a taper leading to the standard Periodogram
             is computed.
        - taper_ft: Fourier transform of the taper. Default as above.
    """
    def __init__(self, spp, freq_min, freq_max, freq_step,
                 minX, maxX, minY, maxY, taper=None, 
                 taper_ft=None):
        super().__init__(freq_min, freq_max, freq_step, minX, maxX, 
                         minY, maxY, taper, taper_ft)
        self.spp = spp
        self.N = len(spp)

        # Define the set of Fourier frequencies
        self.fourier_frequencies = self.freq_set * 2 * np.pi

    @abstractmethod
    def computeSpectralDensity(p, q, **kwargs) -> float:
        """
        Function to evaluate and return the relevant spectral density.
        """
        pass

    def computePeriodogram(self):
        """
        Function to compute the periodogram.
        """
        self.computeSinglePeriodogram(self.spp)

    def computeLikelihood(self, **kwargs):
        """
        Compute and return the Whittle likelihood. Parameters:
            - **kwargs: must contain parameters of the spp at which
                        we want to evaluate the likelihood.      
        """
        # Compute the full periodogram
        self.computePeriodogram()

        # Empty array to store the Whittle likelihood values
        likelihood_eval = np.zeros((len(self.fourier_frequencies) ** 2))

        i = -1; k=-1
        for p in self.freq_set:
            i += 1
            j = -1
            for q in self.freq_set:
                j += 1; k += 1
                spectral_density = self.computeSpectralDensity(
                    p=p, q=q, **kwargs)
                periodogram_eval = self.periodogram[i, j]
                likelihood_eval[k] = (np.log(spectral_density) + 
                                periodogram_eval / spectral_density)

        return (-np.sum(likelihood_eval))

    def scipyOptimisation(
        self, neg_ll, init_params, method='nelder-mead', 
        options={'xatol': 1e-4, 'disp': False}, **kwargs):
        """
        Run the scipy.minimise algorithm with chosen method and options.
        Parameters:

        - neg_ll: a function taking one vector argument that 
                  outputs the negative Whittle lilelihood for
                  use in the optimiser.
        - init_params: vector containing the values at which we initialise
                       the scipy optimiser.
        """
        solution = minimize(neg_ll, init_params, method=method,
                    options=options, **kwargs);

        return solution.x

class ThomasWhittleEstimator(BaseWhittleEstimator):
    """
    A class to compute an estimator based on the Whittle likelihood.
    Parameters:
        - spp: the spatial point process (np.array).
        - freq_min, freq_max, freq_step: integers defining range and 
                step size of the frequencies to evaluate at.
        - minX, maxX, ...: grid we simulate on.
        - taper: normalised taper function. If not supplied, the default is to 
             leave as None and a taper leading to the standard Periodogram
             is computed.
        - taper_ft: Fourier transform of the taper. Default as above.
    """
    def __init__(self, spp, freq_min, freq_max, freq_step,
                 minX, maxX, minY, maxY, taper=None, taper_ft=None):
        super().__init__(spp, freq_min, freq_max, freq_step,minX, maxX, 
                         minY, maxY, taper, taper_ft)

    def computeSpectralDensity(self, rho, K, sigma, p, q):
        """
        Function to compute the spectral density estimate for a Thomas
        process.
        """
        return rho * K * (1 + K * np.exp(-4 * np.pi**2 * (p ** 2 + q ** 2) * (sigma ** 2)))

class LGCPWhittleEstimator(BaseWhittleEstimator):
    """
    A class to compute an estimator based on the Whittle likelihood.
    Parameters:
        - spp: the spatial point process (np.array).
        - freq_min, freq_max, freq_step: integers defining range and 
                step size of the frequencies to evaluate at.
        - minX, maxX, ...: grid we simulate on.
        - taper: normalised taper function. If not supplied, the default is to 
             leave as None and a taper leading to the standard Periodogram
             is computed.
        - taper_ft: Fourier transform of the taper. Default as above.
    """
    def __init__(self, spp, freq_min, freq_max, freq_step,
                 minX, maxX, minY, maxY, taper=None, taper_ft=None):
        super().__init__(spp, freq_min, freq_max, freq_step,minX, maxX, 
                         minY, maxY, taper, taper_ft)
        
        # Override the Fourier frequencies and establish based on value of beta
        self.fourier_frequencies = self.freq_set * 2 * np.pi
        
    def computeSpectralDensity(self, mu, sig2, beta, k_max, p, q):
        """
        Function to compute the spectral density estimate for a LGCP
        with an exponential kernel
        """
        omega = (p ** 2 + q ** 2) ** (1/2)

        if omega * beta < 1 / np.pi:
            kappa = np.pi * omega * beta
            f_trunc = (np.exp(mu + sig2/2) +
                2 * np.pi * np.exp(2 * mu + sig2) * (sig2 ** 2) * (beta ** 2) *
                self._full_sum_to_k(k_max, kappa, sig2)
            )
        else:
            f_trunc = np.nan

        return f_trunc

    def _outer_sum_terms(self, k, kappa):
        """
        Helper function for computeSpectralDensity
        """
    
        @np.vectorize
        def terms_to_k(k_set):
            return (
                ((-1) ** k_set * math.factorial(2*k_set + 1)) / 
                (2**(2*k_set) * (math.factorial(k_set))**2) * 
                kappa ** (2 * k_set)
            )

        k_set = np.arange(k + 1)

        return terms_to_k(k_set)
        
    def _gen_hyper_terms(self, k, sig2):
        """
        Helper function for computeSpectralDensity
        """
        
        k_set = np.arange(k+1)

        gen_hyper_store = np.zeros(k+1)

        for i in k_set:
            a = np.full(2*i + 3, 1); b = np.full(2*i + 3, 2)
            gen_hyper_store[i] = (
                mp.hyper(a, b, sig2)
            )

        return gen_hyper_store 

    def _full_sum_to_k(self, k, kappa, sig2):
        """
        Helper function for computeSpectralDensity
        """

        full_sum = (
            np.cumsum(self._outer_sum_terms(k, kappa) 
                        * self._gen_hyper_terms(k, sig2))
        )

        return full_sum
    
#%%
from spatial_pp import SPP_LGCP
from gstools import Exponential

lgcp = SPP_LGCP(step_size=0.01)
spp = lgcp.simSPP(Exponential, {'dim': 2, 'var': 0.1, 'len_scale': 0.1}, mean=5)
lgcp_w = LGCPWhittleEstimator(spp, minX=-0.5, maxX=0.5, 
                              minY=-0.5, maxY=0.5)

lgcp_w.computeLikelihood(mu=4, sig2=0.15, beta=0.2, k_max=250)