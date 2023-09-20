
import numpy as np
import cmath
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod, abstractstaticmethod

# class BasePeriodogram(ABC):

#     @abstractmethod
#     def _taper(self):
#         pass

#     @abstractmethod
#     def _taper_ft(self):
#         """
#         This must return an array of shape [N,2]
#         """
#         pass

#     @abstractmethod
#     def computeSinglePeriodogram(self):
#         pass

#     @abstractmethod
#     def computeAveragePeriodogram(self):
#         pass

class DebiasedBartlettPeriodogram:
    """
    Class to create the Bartlett periodogram of a 2D spatial point process. Parameters:

    - freq_min, freq_max: minimum and maximum frequencies.
    - freq_step: step size between consecutive frequencies.
    - minX, maxX, ...: coordinates of simulation grid. 
    """
    def __init__(self, freq_min, freq_max, freq_step, minX=-1/2, maxX=1/2, 
                 minY=-1/2, maxY=1/2, p1=1, p2=1):

        self.minX = minX; self.maxX = maxX; self.minY = minY;  self.maxY = maxY
        self.ell_x = maxX - minX; self.ell_y = maxY - minY
        self.area = self.ell_x * self.ell_y
        self.freq_min = freq_min; self.freq_max = freq_max
        self.p1 = p1; self.p2 = p2

        parts = str(freq_step).split('.')
        if len(parts) == 1:
            num_dec_points = 0
        else:
            num_dec_points = len(parts[1])

        # Define the set of frequencies where we want to evaluate the periodogram
        self.freq_set = np.arange(freq_min, freq_max + freq_step, freq_step).round(num_dec_points)

    def _taper(self, x, y, p1, p2):
        return 1 / np.sqrt(self.area)

    def _taper_ft(self, p, q, p1, p2):
        """
        FT of default taper function, to be used if no taper given.
        The FT is (1/\sqrt(|W|)) * \prod_{j=1}^2 sin(\pi * w_j * l_j)/(\pi * w_j) 
        Parameters:
            - p, q: wave function values.
        """
        
        if ((p == 0) & (q == 0)):
            t = np.sqrt(self.area)
        elif ((p == 0) & (q != 0)):
            t = ((1 / np.sqrt(self.area)) * 
                 self.ell_x * np.sin(np.pi * q * self.ell_y) / (np.pi * q)) 
        elif ((p != 0) & (q == 0)):
            # t = np.sin(np.pi * p) * cmath.rect(1, -np.pi * p) / (np.pi * p)
            t = ((1 / np.sqrt(self.area)) * 
                 self.ell_y * np.sin(np.pi * p * self.ell_x) / (np.pi * p))
        else:
            t = ((1 / np.sqrt(self.area)) * 
                 np.sin(np.pi * p * self.ell_x) * np.sin(np.pi * q * self.ell_y) / 
                 (p * q * np.pi ** 2))

        return np.array([t, 0])

    def computeSinglePeriodogram(self, spp, p1=None, p2=None):
        """        
        A method to compute a periodogram for a single point process sample.
        """
        self.spp = spp; self.N = len(self.spp)

        # Estimate lambda
        self.lam_hat = self.N / self.area
        
        # Empty periodogram
        periodogram = np.zeros((len(self.freq_set), len(self.freq_set)))

        # Evaluate the taper
        if p1 is None:
            p1 = self.p1
        if p2 is None:
            p2 = self.p2

        taper_eval = self._taper(self.spp[:,0], self.spp[:,1], p1, p2)

        # Iterate over frequencies
        # Counting index
        p_count = -1
        for p in self.freq_set:
            p_count += 1
            # Scale x-y values by 2pi and by p or q: do p on the 
            # outside of the q loop so it's done once
            spp_freq = self.spp.copy()
            spp_freq[:,0] = self.spp[:,0] * (2 * np.pi * p)
            q_count = -1
            for q in self.freq_set:
                q_count += 1
                # Indices of w=0
                if (p == 0) & (q == 0):
                    p_0 = p_count; q_0 = q_count
                # Scale by 2pi and q
                spp_freq[:,1] = self.spp[:,1] * (2 * np.pi * q)

                # Create array from summing these values (input into trig) - this is
                # -2pi * omega . x
                spp_freq_sum = -(spp_freq[:,0] + spp_freq[:,1])

                # Input the scaled sum into the trig functions, multiply each by the
                # relevant taper and compute the resulting DFT

                # Evaluate the taper FT
                taper_ft_eval = self._taper_ft(p, q, p1, p2)

                # We compute by evaluating the cosine and sine of the arguments, 
                # summing them (real and imaginary) then computing modulus
                cos_sin = np.zeros((self.N, 2))
                cos_sin[:,0] = taper_eval * np.cos(spp_freq_sum)
                cos_sin[:,1] = taper_eval * np.sin(spp_freq_sum) 

                # Debias the peridodogram
                real = cos_sin[:, 0].sum() - self.lam_hat * taper_ft_eval[0]
                imag = cos_sin[:, 1].sum() - self.lam_hat * taper_ft_eval[1]
                
                # Evaluate squared sum of sin and cos terms
                periodogram[q_count, p_count] = real ** 2 + imag ** 2

        self.periodogram = periodogram

    def computeAveragePeriodogram(self, spps, verbose=100):
        """
        Run n_samp simulations and compute periodogram for each, then average.
        Parameters:

            - spps: list of spp from same process.
        """
        n_spp = len(spps)

        sample_periodograms = []

        print("Computing periodograms...")
        for n in range(n_spp):
            if n % verbose == 0:
                print(f"Iteration: {n+1} of {n_spp}")
            spp = spps[n]
            self.computeSinglePeriodogram(spp)
            sample_periodograms.append(self.periodogram)

        # Average the periodograms
        self.average_periodogram = np.mean(np.array(sample_periodograms), axis=0)
        
    def plot(self, average=True, save=False, file_name=None):
        """
        Plot the sampled process and the periodogram.
        """
        if average:
            # Plot the average periodogram
            plt.imshow(self.average_periodogram, 
                    interpolation='nearest', 
                    cmap=plt.cm.viridis, 
                    extent=[self.freq_set[0],
                            self.freq_set[-1],
                            self.freq_set[0],
                            self.freq_set[-1]])
            plt.xlabel(r"$\omega_1$"); plt.ylabel(r"$\omega_2$")
            plt.colorbar()
            if save:
                plt.savefig("./images/" + file_name)
            plt.show()

        else:
            # Plot the periodogram
            plt.imshow(self.periodogram, 
                    interpolation='nearest', 
                    cmap=plt.cm.viridis, 
                    extent=[self.freq_set[0],
                            self.freq_set[-1],
                            self.freq_set[0],
                            self.freq_set[-1]])
            plt.xlabel(r"$\omega_1$"); plt.ylabel(r"$\omega_2$")
            plt.colorbar()
            if save:
                plt.savefig("./images/" + file_name)
            plt.show()


class OrthogSineTaper(DebiasedBartlettPeriodogram):

    def __init__(self, freq_min, freq_max, freq_step, minX=-1/2, maxX=1/2, 
                 minY=-1/2, maxY=1/2, p1=1, p2=1):
        super().__init__(freq_min, freq_max, freq_step, 
                         minX, maxX, minY, maxY, p1, p2)
        
    def _taper(self, x, y, p1, p2):
        return ((2 / (self.ell_x * self.ell_y) ** (1/2)) * 
                np.sin(np.pi * p1 * (x + self.ell_x/2)/self.ell_x) *
                np.sin(np.pi * p2 * (y + self.ell_y/2)/self.ell_y))

    def _taper_ft(self, w1, w2, p1, p2):

        # Shift frequencies slightly if problematic
        if (w1 == p1 / (2 * self.ell_x)) | (w1 == -p1 / (2 * self.ell_x)):
            w1 = w1 + 10 ** -6
            if (w2 == p2 / (2 * self.ell_y)) | (w2 == -p2 / (2 * self.ell_y)):
                w2 = w2 + 10 ** -6

        elif (w2 == p2 / (2 * self.ell_y)) | (w2 == -p2 / (2 * self.ell_y)):
            w2 = w2 + 10 ** -6
            if (w1 == p1 / (2 * self.ell_x)) | (w1 == -p1 / (2 * self.ell_x)):
                w1 = w1 + 10 ** -6

        # Pre-multiplying constant
        K = ((2 / (self.ell_x * self.ell_y) ** (1/2)) * 
            (4 * p1 * p2 * self.ell_x * self.ell_y / 
            ((np.pi ** 2) *
            (p1 ** 2 - 4 * (self.ell_x ** 2) * (w1 ** 2)) *
            (p2 ** 2 - 4 * (self.ell_y ** 2) * (w2 ** 2))
            ))
            )

        # Odd and odd
        if (p1 % 2 == 1) & (p2 % 2 == 1):
            trig_term = np.cos(np.pi * w1 * self.ell_x) * np.cos(np.pi * w2 * self.ell_y)
            return np.array([K * trig_term, 0])
        # Even and even
        elif (p1 % 2 == 0) & (p2 % 2 == 0):
            trig_term = -np.sin(np.pi * w1 * self.ell_x) * np.sin(np.pi * w2 * self.ell_y)
            return np.array([K * trig_term, 0])
        # Odd and even - IMAGINARY
        elif (p1 % 2 == 1) & (p2 % 2 == 0):
            trig_term = np.cos(np.pi * w1 * self.ell_x) * np.sin(np.pi * w2 * self.ell_y)
            return np.array([0, K * trig_term])
        # Even and odd - IMAGINARY
        elif (p1 % 2 == 0) & (p2 % 2 == 1):
            trig_term = np.sin(np.pi * w1 * self.ell_x) * np.cos(np.pi * w2 * self.ell_y)
            return np.array([0, K * trig_term])


class MultiOrthogSineTaper(OrthogSineTaper):

    def __init__(self, freq_min, freq_max, freq_step, minX=-1/2, maxX=1/2, 
                 minY=-1/2, maxY=1/2):
        super().__init__(freq_min, freq_max, freq_step, 
                         minX, maxX, minY, maxY)

    def computeAveragePeriodogram(self, spps, p1, p2, verbose=100):
        """
        Run n_samp simulations and compute periodogram for each, then average.
        Parameters:

            - spps: list of spp from same process.
        """
        n_spp = len(spps)

        sample_periodograms = []

        print("Computing periodograms...")
        for n in range(n_spp):
            if n % verbose == 0:
                print(f"Iteration: {n+1} of {n_spp}")
            spp = spps[n]
            self.computeSinglePeriodogram(spp, p1, p2)
            sample_periodograms.append(self.periodogram)

        # Average the periodograms
        return np.mean(np.array(sample_periodograms), axis=0)
    
    def computeMultitaperSinglePeriodogram(self, spp, P):

        periodogram = np.zeros((P **2, len(self.freq_set), len(self.freq_set)))
        
        index = -1
        for p1 in range(1, P+1):
            for p2 in range(1, P+1):
                index += 1
                # print(f"Taper values: p1={p1}, p2={p2}.")
                self.computeSinglePeriodogram(spp, p1, p2)
                periodogram[index, :, :] = (self.periodogram)

        self.periodogram = np.mean(np.array(periodogram), axis=0) 
        
    def computeMultitaperAveragePeriodogram(self, spps, P):

        average_periodogram = np.zeros((P **2, len(self.freq_set), len(self.freq_set)))
        
        index = -1
        for p1 in range(1, P+1):
            for p2 in range(1, P+1):
                index += 1
                # print(f"Taper values: p1={p1}, p2={p2}.")

                average_periodogram[index, :, :] = (
                    self.computeAveragePeriodogram(spps, p1, p2)
                )

        self.average_periodogram = np.mean(np.array(average_periodogram), axis=0)        
