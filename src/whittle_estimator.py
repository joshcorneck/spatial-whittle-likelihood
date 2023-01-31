import numpy as np
import scipy, time
from scipy.optimize import minimize
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Union

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
    def __init__(self, spp, freq_min=-16, freq_max=16, freq_step=1,
                 minX=0, maxX=1, minY=0, maxY=1, taper=None, taper_ft=None):
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
    def __init__(self, spp, freq_min=-16, freq_max=16, freq_step=1,
                 minX=0, maxX=1, minY=0, maxY=1, taper=None, taper_ft=None):
        super().__init__(spp, freq_min, freq_max, freq_step,minX, maxX, 
                         minY, maxY, taper, taper_ft)

    def computeSpectralDensity(self, rho, K, sigma, p, q):
        """
        Function to compute the spectral density estimate for a Thomas
        process.
        """
        return rho * K * (1 + K * np.exp(-4 * np.pi**2 * (p ** 2 + q ** 2) * (sigma ** 2)))

    # def gradient(self, rho, K, sigma):
    #     """
    #     The gradient of the Whittle likelihood.
    #     """
    #     # Empty array to store the derivative terms prior to sum
    #     grad_rho = np.zeros((len(self.fourier_frequencies) ** 2))
    #     grad_K = np.zeros((len(self.fourier_frequencies) ** 2))
    #     grad_sig = np.zeros((len(self.fourier_frequencies) ** 2))

    #     i = -1
    #     for p in self.fourier_frequencies:
    #         for q in self.fourier_frequencies:
    #             i += 1
    #             # Compute omega
    #             omega = np.sqrt(p ** 2 + q ** 2)
    #             # Compute f(omega; theta)
    #             spectral_density = self.computeSpectralDensity(
    #                 rho=rho, K=K, sigma=sigma, p=p, q=q)
    #             # Compute I(omega)
    #             periodogram = self.computePeriodogram(p=p, q=q)
    #             # Compute pdv terms
    #             exp_term = -(K * omega) ** 2
    #             pdv_f_rho = K + (K **2) * np.exp(exp_term)
    #             pdv_f_K = rho + 2 * rho * K * np.exp(exp_term)
    #             pdv_f_sig = -2 * rho * sigma * ((K * omega) ** 2) * np.exp(exp_term)
    #             # Populate the gradient terms
    #             outside_factor = (periodogram/(spectral_density ** 2) - 1/spectral_density)
    #             grad_rho[i] = outside_factor * pdv_f_rho
    #             grad_K[i] = outside_factor * pdv_f_K
    #             grad_sig[i] =  outside_factor * pdv_f_sig

    #     return np.array([np.sum(grad_rho), np.sum(grad_K), np.sum(grad_sig)])


    # def gradientAscent(self, init_params, learn_rate=0.1, n_iter=50, tolerance=1e-02):
    #     """
    #     Gradient ascent algorithm for maximising the Whittle lieklihood.
    #     """
    #     # Initializing the values of the variables
    #     param_vector = np.array(init_params)

    #     # Setting up and checking the learning rate
    #     learn_rate = np.array(learn_rate)

    #     # Performing the gradient descent loop
    #     for _ in range(n_iter):
    #         # Recalculating the difference
    #         diff = learn_rate * np.array(self.gradient(param_vector[0],
    #                                                    param_vector[1],
    #                                                    param_vector[2]))

    #         # Checking if the absolute difference is small enough
    #         if np.all(np.abs(diff) <= tolerance):
    #             break

    #         # Updating the values of the variables
    #         param_vector += diff

    #     return param_vector if param_vector.shape else param_vector.item()

