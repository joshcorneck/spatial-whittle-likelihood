#%%
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Periodogram:
    """
    Class to create the periodogram of a 2D spatial point process. Parameters:

    - spp: an initialised class for simulating a 2D SPP
    - freq_min, freq_max: minimum and maximum frequencies 
    - freq_step: step size between consecutive frequencies
    """
    def __init__(self, spp, freq_min, freq_max, freq_step):

        self.spp = spp
        self.freq_min = freq_min; self.freq_max = freq_max

        # Define the set of frequencies where we want to evaluate the periodogram
        self.freq_set = np.arange(freq_min, freq_max + 1, freq_step)

    def computePeriodogram(self, normalise=False, **kwargs) :

        # Get a realisation of the point process we consider
        self.sampledSPP = self.spp.simSPP(**kwargs)
        N = len(self.sampledSPP)

        # Empty periodogram
        periodogram = np.zeros((len(self.freq_set), len(self.freq_set)))

        # Iterate over frequencies
        # Counting index
        p_count = -1
        for p in self.freq_set:
            p_count += 1
            # Scale x-y values by 2pi and by p or q: do p on the 
            # outside of the q loop so it's done once
            spp_freq = self.sampledSPP.copy()
            spp_freq[:,0] = self.sampledSPP[:,0] * 2*np.pi*p
            q_count = -1
            for q in self.freq_set:
                q_count += 1
                # Scale by 2pi and q
                spp_freq[:,1] = self.sampledSPP[:,1] * 2*np.pi*q

                # Create array from summing these values (input into trig)
                spp_freq_sum = spp_freq[:,0] + spp_freq[:,1]

                # Input the scaled sum into the trig functions and compute periodogram
                cos_sin = np.zeros((N,2))
                cos_sin[:,0] = np.cos(spp_freq_sum); cos_sin[:,1] = np.sin(spp_freq_sum)
                if (p == 0) & (q == 0):
                    periodogram[p_count, q_count] = 0
                else:
                    periodogram[p_count,q_count] = np.sum(np.sum(cos_sin, axis=0) ** 2)

        # Normalise the array
        if normalise:
            max_val = np.amax(periodogram)
            self.periodogram = periodogram / max_val
        else:
            self.periodogram = periodogram

    def averagePeriodogram(self, n_samp, normalise=True, **kwargs):
        """
        Run n_samp simulations and compute periodogram for each, then average
        """
        sample_periodograms = []

        for n in range(n_samp):
            if n/100 == np.floor(n/100):
                print(f"Computing sample: {n}")
            self.computePeriodogram(**kwargs)
            sample_periodograms.append(self.periodogram)

        # Average the periodograms
        average_periodogram = np.mean(np.array(sample_periodograms), axis=0)

        if normalise:
            max_val = np.amax(average_periodogram)
            self.average_periodogram = average_periodogram / max_val
        else:
            self.average_periodogram = average_periodogram


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
            plt.scatter(self.sampledSPP[:,0], self.sampledSPP[:,1], 
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

pois = SPP_HomPoisson()
thom = SPP_Thomas()
#%%
per = Periodogram(thom, -10, 10, 1)
per.averagePeriodogram(n_samp = 1000, kappa=50, alpha=25, sigma=0.03, 
                       cov=np.array([[1,0], [0, 1]]), enlarge=1.25)
per.plot()
#%%
per = Periodogram(thom, -10, 10, 1)
per.computePeriodogram(normalise=True, kappa=50, alpha=25, sigma=0.03, 
                       cov=np.array([[1,-0.5], [-0.5, 1]]), enlarge=1.25)
per.plot(average=False)
#%%
per = Periodogram(pois, -10, 10, 1)
per.averagePeriodogram(n_samp=1000, lambdaHom=100)
per.plot()

#%%
# Analytic periodogram - 

def analytic_periodogram(p, q, sigma, kappa, alpha):
    omega = np.sqrt(p**2 + q**2)
    return (kappa * alpha * (1 + alpha * 
            np.exp(-(omega ** 2) * (sigma ** 2))))

p_set = np.arange(-10, 11, 1); q_set = np.arange(-10, 11, 1)
kappa=50; alpha=25; sigma=0.03

periodogram = np.zeros((len(p_set), len(q_set)))

# Iterate over frequencies
# Counting index
p_count = -1
for p in p_set:
    p_count += 1
    q_count = -1
    for q in q_set:
        q_count += 1
        periodogram[p_count, q_count] = analytic_periodogram(p, q, sigma, kappa, alpha)

max_val = np.amax(periodogram)
periodogram = periodogram / max_val

plt.imshow(periodogram, 
            interpolation='nearest', 
            cmap=plt.cm.viridis, 
            extent=[-10,
                    10,
                    -10,
                    10])
plt.colorbar()
plt.show()
# %%
