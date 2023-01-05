#%%
import numpy as np

from spatial_pp import SPP_Thomas

class ThomasWhittleEstimator:
    """
    A class to compute an estimator based on the Whittle likelihood.
    Parameters:
        - spp: the spatial point process (np.array)
    """
    def __init__(self, spp, freq_min, freq_max, freq_step):
        self.spp = spp
        self.N = len(spp)
        self.freq_min = freq_min; self.freq_max = freq_max

        # Define the set of frequencies where we want to evaluate the periodogram
        self.freq_set = np.arange(freq_min, freq_max + 1, freq_step)

    @staticmethod
    def ThomasSpectralDensity(rho, K, sigma, p, q):
        """
        Function to compute the spectral density estimate for a Thomas
        process.
        """
        return rho * K * (1 + K * np.exp(-(p ** 2 + q ** 2) * (sigma ** 2)))

    def computePeriodogram(self, p, q):

        spp_freq = self.spp.copy()
        spp_freq[:,0] = self.spp[:,0] * 2*np.pi*p

        # Scale by 2pi and q
        spp_freq[:,1] = self.spp[:,1] * 2*np.pi*q

        # Create array from summing these values (input into trig)
        spp_freq_sum = spp_freq[:,0] + spp_freq[:,1]

        # Input the scaled sum into the trig functions and compute periodogram
        cos_sin = np.zeros((self.N,2))
        cos_sin[:,0] = np.cos(spp_freq_sum); cos_sin[:,1] = np.sin(spp_freq_sum)

        periodogram = np.sum(np.sum(cos_sin, axis=0) ** 2)

        return periodogram

    def computeLikelihood(self, rho, K, sigma):
        """
        Compute and return the Whittle likelihood        
        """
        # Define the set of Fourier frequencies
        fourier_frequencies = self.freq_set * 2 * np.pi

        # Empty array to store the Whittle likelihood values
        likelihood = np.zeros((len(fourier_frequencies) ** 2))
        
        i = -1
        for p in fourier_frequencies:
            for q in fourier_frequencies:
                i += 1
                spectral_density = self.ThomasSpectralDensity(rho=rho, K=K, sigma=sigma, p=p, q=q)
                periodogram = self.computePeriodogram(p=p, q=q)
                likelihood[i] = np.log(spectral_density) + periodogram / spectral_density

        return (-np.sum(likelihood))

    def maximiseWhittleLikelihood(self):
        pass

#%%
tom = SPP_Thomas()
spp = tom.simSPP(50, 25, 0.03, np.array([[1,0], [0, 1]]), 1.25)

twe = ThomasWhittleEstimator(spp, -10, 10, 1)
twe.computeLikelihood(50, 25, 0.01)
#%%
twe.freq_set