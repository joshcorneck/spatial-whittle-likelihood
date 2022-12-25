#%%

from astropy.stats import RipleysKEstimator
import numpy as np
from scipy.integrate import trapezoid

from spatial_pp import SPP_Thomas

class MinimumContrastThomas:
    """
    A class to implement minimum contrast estimation for a Thomas process with
    parameter theta. Parameters:

    - minX, maxX, minY, maxY: boundaries of grid.
    - spp: class for simulating thomas point process (parameters passed in)
    """
    def __init__(self, thomas_spp, kappa, alpha, sigma, enlarge, 
                 minX=0, maxX=1, minY=0, maxY=1):
        self.minX = minX; self.maxX = maxX; self.minY = minY; self.maxY = maxY
        self.kappa = kappa; self.alpha = alpha; self.sigma = sigma; self.enlarge = enlarge
        self.thomas_spp = thomas_spp
    
    def KfunctionEstimate(self):
        """
        Empirical estimator of Ripley's K-function with a user-specified edge 
        correction.
        """
        # Simulate a Thomas process
        self.point_pattern = self.thomas_spp.simSPP(kappa=self.kappa, alpha=self.alpha, 
                                                    sigma=self.sigma, enlarge=self.enlarge)
        self.N = len(self.point_pattern)

        # r values to estimate over
        self.r_range = np.linspace(self.minX + 0.01, self.maxX/4, 100)

        # Estimate the K-function 
        Kest = RipleysKEstimator(area=1, x_min=self.minX, x_max=self.maxX, 
                                 y_min=self.minY, y_max=self.maxY)
        self.point_pattern = self.point_pattern.reshape((self.N, 2))
        self.Kfun_est = Kest(data=self.point_pattern, radii=self.r_range, mode='ripley')

    @staticmethod
    def analyticKfun(r, kappa, sigma):
        analyticKfun = (
            np.pi * r ** 2 + (1 / kappa) * (1 - np.exp(- (r ** 2)/(4 * sigma ** 2))))

        return analyticKfun

    def minimumContrastEstimate(self, p=2):

        # Run a simulation and estimate its K-function
        self.KfunctionEstimate()

        # For a given grid of values, compute the analytic K function 
        # and compute difference between it and the estimate raised 
        # to p
        kappa_grid = np.linspace(20, 30, 10); N_kappa = len(kappa_grid)
        sigma_grid = np.linspace(0.01, 0.1, 10); N_sigma = len(sigma_grid)

        integral_grid = np.zeros((N_kappa, N_sigma))

        @staticmethod
        def Kfun_difference(kappa, sigma):
            return abs(
                self.Kfun_est - 
                self.analyticKfun(self.r_range, kappa, sigma)
                ) ** 2

        @staticmethod
        def integrate(kappa, sigma):
            return (trapezoid(Kfun_difference(kappa, sigma),
                              self.r_range))

        for i in range(N_kappa):
            kappa = kappa_grid[i]
            for j in range(N_sigma):
                sigma = sigma_grid[j]
                integral = integrate(kappa, sigma)
                integral_grid[i,j] = integral

        min_value = np.min(integral_grid)
        kappa_min = kappa_grid[np.where(integral_grid == min_value)[0][0]]
        sigma_min = sigma_grid[np.where(integral_grid == min_value)[1][0]]
        alpha_min = self.N / kappa_min

        return np.array([kappa_min, alpha_min, sigma_min])

#%%
spp_t = SPP_Thomas()
mct = MinimumContrastThomas(spp_t, kappa=25, alpha=50, sigma=0.03, enlarge=1.25)
mct.minimumContrastEstimate()

# %%

