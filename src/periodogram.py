#%%
import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Periodogram:
    """
    Class to create the periodogram of a 2D spatial point process. Parameters:

    - freq_min, freq_max: minimum and maximum frequencies.
    - freq_step: step size between consecutive frequencies.
    - minX, maxX, ...: coordinates of simulation grid. 
    - taper: normalised taper function. If not supplied, the default is to 
             leave as None and a taper leading to the standard Periodogram
             is computed.
    - taper_ft: Fourier transform of the taper. Default as above.
    """
    def __init__(self, freq_min, freq_max, freq_step, minX=0, maxX=1, 
                 minY=0, maxY=1, taper=None, taper_ft=None):

        self.minX = minX; self.maxX = maxX; self.minY = minY;  self.maxY = maxY
        self.area = (maxX - minX) * (maxY - minY)

        # If no taper supplied, give the default leading to periodogram
        if taper is None:
            print("Defaulting to taper leading to standard periodogram.")
            self.default = True
        else: 
            self.default = False
            if taper_ft is None:
                raise Exception("taper_ft not supplied.")
            else:
                self.taper = taper; self.taper_ft = taper_ft

        self.freq_min = freq_min; self.freq_max = freq_max

        # Define the set of frequencies where we want to evaluate the periodogram
        self.freq_set = np.arange(freq_min, freq_max + 1, freq_step)

    def taper_default(self):
        """
        Default taper function, to be used if no taper given.
        """
        return 1/np.sqrt(self.area)

    def taper_ft_default(self, p, q):
        """
        FT of default taper function, to be used if no taper given. Parameters:

        - p, q: wave function values.
        """
        
        if ((p == 0) & (q == 0)):
            t =  np.sqrt(self.area)
        elif ((p == 0) & (q != 0)):
            t = np.sin(np.pi * q) * cmath.rect(1, -np.pi * q) / (np.pi * q)
        elif ((p != 0) & (q == 0)):
            t = np.sin(np.pi * p) * cmath.rect(1, -np.pi * p) / (np.pi * p)
        else:
            t = (np.sin(np.pi * p) * np.sin(np.pi * q) *
                cmath.rect(1, -np.pi * p - np.pi * q) /
                (p * q * np.pi ** 2))

        t = t * (1 / np.sqrt(self.area))

        return np.array([t.real, t.imag])

    def computeSinglePeriodogram(self, spp):

        self.spp = spp; self.N = len(self.spp)

        # Estimate lambda
        self.lam_hat = self.N / self.area
        
        # Empty periodogram
        periodogram = np.zeros((len(self.freq_set), len(self.freq_set)))

        # Iterate over frequencies
        # Counting index
        p_count = -1
        for p in self.freq_set:
            p_count += 1
            # Scale x-y values by 2pi and by p or q: do p on the 
            # outside of the q loop so it's done once
            spp_freq = self.spp.copy()
            spp_freq[:,0] = self.spp[:,0] * 2*np.pi*p
            q_count = -1
            for q in self.freq_set:
                q_count += 1
                # Scale by 2pi and q
                spp_freq[:,1] = self.spp[:,1] * 2*np.pi*q

                # Create array from summing these values (input into trig) - this is
                # 2pi * omega . x
                spp_freq_sum = spp_freq[:,0] + spp_freq[:,1]

                # Input the scaled sum into the trig functions, multiply each by the
                # relevant taper and compute the resulting DFT

                # If using default taper, set accordingly
                if self.default:
                    # Evaluate taper function
                    taper_eval = self.taper_default()

                    # Evaluate Fourier transform of taper
                    taper_ft_eval = self.taper_ft_default(p, q)
                else:
                    # Evaluate taper function
                    taper_eval = self.taper(self.spp)

                    # Evaluate Fourier transform of taper
                    taper_ft_eval = self.taper_ft(p, q)

                cos_sin = np.zeros((self.N,2))
                cos_sin[:,0] = taper_eval * np.cos(spp_freq_sum)
                cos_sin[:,1] = taper_eval * np.sin(spp_freq_sum) 

                real = cos_sin[:, 0].sum() - self.lam_hat * taper_ft_eval[0]
                imag = cos_sin[:, 1].sum() - self.lam_hat * taper_ft_eval[1]
                
                # Evaluate squared sum of sin and cos terms
                periodogram[p_count,q_count] = real ** 2 + imag ** 2

        self.periodogram = periodogram

    def computeAveragePeriodogram(self, spps):
        """
        Run n_samp simulations and compute periodogram for each, then average.
        Parameters:

            - spps: list of spp from same process.
        """
        n_spp = len(spps)

        sample_periodograms = []

        for n in range(n_spp):
            spp = spps[n]
            self.computeSinglePeriodogram(spp)
            sample_periodograms.append(self.periodogram)

        # Average the periodograms
        self.average_periodogram = np.mean(np.array(sample_periodograms), axis=0)

    def computePeriodogram(self, spps):

        # Check if spps is list and if so run appropriate method
        if isinstance(spps, list):
            if len(spps) == 1:
                self.computePeriodogram(spps[0])
            else:
                self.computeAveragePeriodogram(spps)
        else:
            raise Exception("spp must be a list")
        
    def plot(self, average=True):
        """
        Plot the sampled process and the periodogram.
        """
        if average:
            # Plot the average periodogram
            plt.imshow(self.average_periodogram, 
                    interpolation='nearest', 
                    cmap=plt.cm.viridis, 
                    extent=[self.freq_min,
                            self.freq_max,
                            self.freq_min,
                            self.freq_max])
            plt.colorbar()
            plt.show()
        else:
            # Plot the process realisation
            plt.scatter(self.spp[:,0], self.spp[:,1], 
                        edgecolor='b', alpha=0.5)
            plt.xlabel("x"); plt.ylabel("y")
            plt.show()
            plt.clf()

            # Plot the periodogram
            plt.imshow(self.periodogram, 
                    interpolation='nearest', 
                    cmap=plt.cm.viridis, 
                    extent=[self.freq_min,
                            self.freq_max,
                            self.freq_min,
                            self.freq_max])
            plt.colorbar()
            plt.show()

#%% 
from spatial_pp import SPP_HomPoisson, SPP_Thomas
#%%
per = Periodogram(thom, -10, 10, 1)
per.averagePeriodogram(n_samp = 1000, kappa=50, alpha=25, sigma=0.03, 
                       cov=np.array([[1,0], [0, 1]]), enlarge=1.25)
per.plot()
#%%
tom = SPP_Thomas()
spp = tom.simSPP(50, 25, 0.01, np.array([[1, 0], [0, 1]]), enlarge=1.25)

per = Periodogram(-16, 16, 1)
per.computeSinglePeriodogram(spp)
#%%
spps = []
for j in range(1000):
    spps.append(tom.simSPP(50, 25, 0.01, np.array([[1, 0], [0, 1]]), enlarge=1.25))

#%%
per = Periodogram(-16, 16, 1)
per.computeAveragePeriodogram(spps)
#%%
