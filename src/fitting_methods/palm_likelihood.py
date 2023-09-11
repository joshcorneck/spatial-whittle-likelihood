import numpy as np

from spatial_pp import SPP_Thomas

class PalmLikelihoodThomas:
    """
    A class to estimate the parameters of a Thomas point process 
    by maximising the Palm likelihood for the process. Parameters:
    
        - spp: a point pattern of a Thomas proces (np.array of shape N x 2)
        - R: assumed maximum range of correlation of the process.
    """
    def __init__(self, spp, R, minX=0, maxX=1, minY=0, maxY=1):
        self.minX = minX; self.maxX = maxX; self.minY = minY; self.maxY = maxY
        self.spp = spp
        self.R = R

        # Number of points in the process
        self.N = len(spp)

    def computeFryPattern(self):

        fry_pattern = np.zeros((self.N*(self.N-1), 2))

        # Counter for indexing
        k = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                fry_pattern[k, :] = self.spp[i, :] - self.spp[j, :]
                k += 1

        # Second half is just negative of the first (ordered pairs)
        fry_pattern[int(self.N*(self.N-1)/2):(self.N*(self.N-1)), :] = -(
            fry_pattern[0:int(self.N*(self.N-1)/2), :]
        )

        return fry_pattern

    def computePalmLikelihood(self, less_R, rho, K, sigma):
        """
        Parameters:

            - less_R: the distances between points that are less than the assumed
                      maximum range of correlation.
        """
        # Compute the likelihood
        log_palm = (
            np.sum(
                np.log(
                    K * (rho + 1/(4*np.pi*sigma**2) * np.exp(-less_R ** 2/(4 * sigma ** 2)))
                )
            ) 
            -
            self.N * K * (np.pi * rho * self.R ** 2 + 1 - np.exp(-self.R**2/(4*sigma**2)))
        )

        return log_palm

    def maximisePalmLikelihood(
        self, 
        rho_min, rho_max, K_min, K_max, sig_min, sig_max, 
        coarseness, random, verbose):
        """
        Method to compute Whittle likelihood over grid of values and 
        to return the maximisers. Parameters:

        - rho_min, rho_max, ...: determine range of values for random search of
                             each parameter.
        - coarseness: (cube root of) how many equally spaced segments we divide 
                       the random search grid into.
        - random: Boolean for whether random search or not.
        - verbose: Boolean for whether to print progress.
        """

        if random:
            # Parameters for random optimisation
            rho_set = np.random.uniform(rho_min, rho_max, coarseness)
            K_set = np.random.uniform(K_min, K_max, coarseness)
            sig_set = np.random.uniform(sig_min, sig_max, coarseness)
        else:
            # Parameters for grid optimisation
            rho_set = np.linspace(rho_min, rho_max, coarseness)
            K_set = np.linspace(K_min, K_max, coarseness)
            sig_set = np.linspace(sig_min, sig_max, coarseness)

        # Empty grid to store 
        Palm_grid = np.zeros((coarseness, coarseness, coarseness))

        # Compute the distances between points and pick those less than R. Tjese
        # are what we input into the Palm likelihood function.
        fry_pattern = self.computeFryPattern()
        dists = (fry_pattern[:,0] ** 2 + fry_pattern[:, 1] ** 2) ** (1/2)
        less_R = dists[dists < self.R]

        count = 0        
        for i, rho in enumerate(rho_set):
            for j, K in enumerate(K_set):
                for k, sig in enumerate(sig_set):
                    if verbose:
                        if ((count / 100) == (count // 100)):
                            print(f"Iteration {count + 1} of {coarseness ** 3}")
                    count += 1
                    Palm_grid[i,j,k] = (
                        self.computePalmLikelihood(less_R, rho, K, sig)
                    )

        # Index of maximum value in flattened array
        amax = Palm_grid.argmax()
        max_idx = np.where(Palm_grid == Palm_grid.flatten()[amax])

        # Maximum values
        rho_max = rho_set[max_idx[0][0]]
        K_max = K_set[max_idx[1][0]]
        sig_max = sig_set[max_idx[2][0]]

        return np.array([rho_max, K_max, sig_max])
    



