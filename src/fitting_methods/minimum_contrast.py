#%%
from astropy.stats import RipleysKEstimator
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize

class MinimumContrastThomas:
    """
    A class to implement minimum contrast estimation for a Thomas process with
    parameter theta. Parameters:

    - minX, maxX, minY, maxY: boundaries of grid.
    - spp: class for simulating thomas point process (parameters passed in)
    """
    def __init__(self, thomas_spp, minX=0, maxX=1, minY=0, maxY=1):
        self.minX = minX; self.maxX = maxX; self.minY = minY; self.maxY = maxY
        self.thomas_spp = thomas_spp
    
    def KfunctionEstimate(self):
        """
        Empirical estimator of Ripley's K-function with a user-specified edge 
        correction.
        """
        self.N = len(self.thomas_spp)

        # r values to estimate over
        self.r_range = np.linspace(self.minX, self.maxX/8, 100)

        # Estimate the K-function 
        Kest = RipleysKEstimator(area=1, x_min=self.minX, x_max=self.maxX, 
                                 y_min=self.minY, y_max=self.maxY)
        self.thomas_spp = self.thomas_spp.reshape((self.N, 2))
        self.Kfun_est = Kest(data=self.thomas_spp, radii=self.r_range, mode='none')

    @staticmethod
    def analyticKfun(r, rho, sigma):
        analyticKfun = (
            np.pi * r ** 2 + (1 / rho) * (1 - np.exp(- (r ** 2)/(4 * sigma ** 2))))

        return analyticKfun

    def scipyOptimisation(
        self, init_params, c=0.25, method='nelder-mead', 
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
        # Run a simulation and estimate its K-function
        self.KfunctionEstimate()

        @staticmethod
        def Kfun_difference(rho, sigma):
            return abs(
                self.Kfun_est ** c - 
                self.analyticKfun(self.r_range, rho, sigma) ** c
                ) ** 2

        @staticmethod
        def integrate(rho, sigma):
            return (trapezoid(Kfun_difference(rho, sigma), self.r_range))

        @staticmethod
        def minimise_fnc(x):
            return integrate(x[0], x[1])

        solution = minimize(minimise_fnc, init_params, method=method,
                    options=options, **kwargs);

        params = [solution.x[0], self.N/solution.x[0], solution.x[1]]

        return params

    def minimumContrastEstimate(self, rho_grid, sigma_grid, c=0.25):

        # Run a simulation and estimate its K-function
        self.KfunctionEstimate()

        # For a given grid of values, compute the analytic K function to the c
        # and compute the squared difference between it and the estimate also
        # to the power of c
        N_rho = len(rho_grid); N_sigma = len(sigma_grid)

        integral_grid = np.zeros((N_rho, N_sigma))

        @staticmethod
        def Kfun_difference(rho, sigma):
            return abs(
                self.Kfun_est ** c - 
                self.analyticKfun(self.r_range, rho, sigma) ** c
                ) ** 2

        @staticmethod
        def integrate(rho, sigma):
            return (trapezoid(Kfun_difference(rho, sigma), self.r_range))

        for i in range(N_rho):
            rho = rho_grid[i]
            for j in range(N_sigma):
                sigma = sigma_grid[j]
                integral = integrate(rho, sigma)
                integral_grid[i,j] = integral

        min_value = np.min(integral_grid)
        rho_min = rho_grid[np.where(integral_grid == min_value)[0][0]]
        sigma_min = sigma_grid[np.where(integral_grid == min_value)[1][0]]
        K_min = self.N / rho_min

        return np.array([rho_min, K_min, sigma_min])

#%%

# import matplotlib.pyplot as plt

# spp_t = SPP_Thomas()
# mct = MinimumContrastThomas(spp_t, rho=50, K=15, sigma=0.03, 
#                             cov = np.array([[1,0], [0, 1]]), enlarge=1.25)
# # samp_params = mct.scipyOptimisation(init_params=[15, 0.05])
# samp_params = mct.minimumContrastEstimate(rho_grid=np.linspace(10,100,100),
#                                           sigma_grid=np.linspace(0.01,0.1,100))
# samp_params
#%%

# plt.figure(1);
# plt.scatter(np.log(params[:,0]), np.log(params[:,1]));
# plt.xlabel(r"$\log(\hat{\rho})$"); plt.ylabel(r"$\log(\hat{\sigma})$");

# data = [(params[:,0]- 25)/25, (params[:,1]- 0.03)/0.03]
# fig7, ax7 = plt.subplots()
# ax7.set_title('Normalised differnece between parameter estimates and true values')
# ax7.boxplot(data)
# plt.xticks([1, 2], [r'$(\hat{\rho} - \rho)/\rho$', r'$(\hat{\sigma} - \sigma)/\sigma$'])
# plt.show()

# plt.figure(2);
# plt.boxplot((params[:,0]- 25)/25);
# plt.xticks([1], ['']); plt.title(r"Plot of $(\hat{\rho} - \rho)/\rho$")
# plt.figure(3);
# plt.boxplot((params[:,1]- 0.03)/0.03);
# plt.xticks([1], ['']); plt.title(r"Plot of $(\hat{\sigma} - \sigma)/\sigma$")

# %%
