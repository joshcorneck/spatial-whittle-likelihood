from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import math
import mpmath as mp

class BaseWhittleEstimator(ABC):
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
    def __init__(self, spp, periodogram, freq_min, freq_max, freq_step, 
                 minX, maxX, minY, maxY):

        self.spp = spp; self.N = len(spp); self.periodogram = periodogram
        self.minX = minX; self.maxX = maxX; self.minY = minY;  self.maxY = maxY
        self.ell_x = maxX - minX; self.ell_y = maxY - minY
        self.area = self.ell_x * self.ell_y
        self.freq_min = freq_min; self.freq_max = freq_max

        # Define the set of frequencies where we want to evaluate the periodogram
        self.freq_set = np.arange(freq_min, freq_max + freq_step, freq_step)

        # Define the set of Fourier frequencies
        self.fourier_frequencies = self.freq_set * 2 * np.pi

    @abstractmethod
    def computeSpectralDensity(p, q, **kwargs) -> float:
        """
        Function to evaluate and return the relevant spectral density.
        """
        pass

    def computeLikelihood(self, omega_skip=None, **kwargs):
        """
        Compute and return the Whittle likelihood. Parameters:
            - periodogram: np.array of the periodogram.
            - omega_skip: values of omega to avoid in the inference. This
                          should be a np.array of shape (K,2), with K
                          being the number of frequencies we need to skip.
            - **kwargs: must contain parameters of the spp at which
                        we want to evaluate the likelihood.      
        """
        # Check likelihood shape matches inputted frequency range
        if self.periodogram.shape != (len(self.freq_set), len(self.freq_set)):
            raise ValueError("""Shape of inputted periodogram doesn't match the
                             given frequency range.""")

        # Empty array to store the Whittle likelihood values
        likelihood_eval = np.zeros((len(self.fourier_frequencies) ** 2))

        i = -1; k=-1
        for p in self.freq_set:
            i += 1
            j = -1
            for q in self.freq_set:
                j += 1; k += 1
                # Experimentation shows problems approx within circle or radius 1
                # of the removeable pole. Check if p and q fall in this circle. If
                # they do, keep likelihood at problematic values 0.
                valid_freq = True
                if ((p == 0) & (q == 0)):
                    valid_freq = False
                if valid_freq:   
                    spectral_density = self.computeSpectralDensity(p=p, q=q, **kwargs)
                    periodogram_eval = self.periodogram[j, i]
                    likelihood_eval[k] = -(np.log(spectral_density) + 
                                    periodogram_eval / spectral_density)

        return np.sum(likelihood_eval)

    def scipyOptimisation(self, neg_ll, init_params, method=None, **kwargs):
        """
        Run the scipy.minimise algorithm with chosen method and options.
        Parameters:

        - neg_ll: a function taking one vector argument that 
                  outputs the negative Whittle lilelihood for
                  use in the optimiser.
        - init_params: vector containing the values at which we initialise
                       the scipy optimiser.
        """
        if method is None:
            solution = minimize(neg_ll, init_params, **kwargs);
        else:
            solution = minimize(neg_ll, init_params, method=method, **kwargs);

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
    def __init__(self, spp, periodogram, freq_min, freq_max, freq_step,
                 minX, maxX, minY, maxY):
        super().__init__(spp, periodogram, freq_min, freq_max, freq_step,minX, maxX, 
                         minY, maxY)

    def computeSpectralDensity(self, lam_hat, rho_tilde, sig_tilde, p, q):
        """
        Function to compute the spectral density estimate for a Thomas
        process. Note that this is now parameterised entirely in terms of 
        the transformed variables.
        """
        return lam_hat * (1 + lam_hat * np.exp(-(4 * np.pi ** 2 * (p ** 2 + q ** 2) *
                                                 np.exp(2 * sig_tilde) + rho_tilde)))
    
    def computeLikelihoodDerivative(self, lam_hat, rho_tilde, sig_tilde):
        """
        This will compute the full likelihood derivative. This will be parameterised in
        terms of the transformed variables, and is the derivative with respect to 
        those variables.
        """
        # Empty array to store the derivative frequency evaluations
        deriv_eval = np.zeros((2, len(self.fourier_frequencies) ** 2))

        i = -1; k=-1
        for p in self.freq_set:
            i += 1
            j = -1
            for q in self.freq_set:
                j += 1; k += 1
                spec_density = self.computeSpectralDensity(lam_hat, rho_tilde, sig_tilde, p, q)
                spec_deriv = self._spectralDerivative(lam_hat, rho_tilde, sig_tilde, p, q)
                periodogram_eval = self.periodogram[j, i]
                deriv_eval[:,k] = - spec_deriv / spec_density * (1 - periodogram_eval / spec_density)

        gradients = deriv_eval.sum(axis=1)

        return gradients

    def _spectralDerivative(self, lam_hat, rho_tilde, sig_tilde, p, q):
        """
        The derivative of f(omega;theta_tilde) with respect to rho_tilde and sig_tilde. 
        Note that this is parameterised in terms of the transformed variables.
        """
        der_rho_tilde = -(lam_hat ** 2 * np.exp(-4 * np.pi ** 2 * (p ** 2 + q ** 2) * 
                                                  np.exp(2 * sig_tilde) - rho_tilde)
        )
        der_sig_tilde = -(8 * np.pi ** 2 * (p ** 2 + q ** 2) * lam_hat ** 2 *
                          np.exp(2 * sig_tilde - rho_tilde - 4 * np.pi ** 2 * (p ** 2 + q ** 2) * 
                                 np.exp(2 * sig_tilde)))
        
        return np.array([der_rho_tilde, der_sig_tilde])


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
