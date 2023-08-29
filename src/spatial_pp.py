"""
This document contains classes for sampling from different point processes.
Point processes that can be simulated are:
    - Homogeneous Poisson
    - Inhomogeneous Poisson
    - Thomas 
    - LGCP
"""
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from gstools import SRF, Exponential

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod, abstractstaticmethod

class SPP(ABC):
    """
    Abstract base class for spatial point processes
    """
    @abstractmethod
    def simSPP(self) -> np.array:
        pass

    @abstractstaticmethod
    def plot():
        pass

    @abstractmethod
    def sample_and_plot(self):
        pass


class SPP_HomPoisson(SPP):
    """
    A class to simulate a homogeneous Poisson process on 
    a rectangular window. Parameters:

    - minX, maxX, minY, maxY: end coordinates of the simulation window
    """
    def __init__(self, minX=0, maxX=1, minY=0, maxY=1):
        self.minX = minX; self.maxX = maxX 
        self.minY = minY; self.maxY = maxY 

        # Compute parameters of the simulation window
        self.lenX = maxX - minX; self.lenY = maxY - minY
        self.area = self.lenX * self.lenY

    def simSPP(self, lambdaHom, enlarge=1) -> np.array:
        """
        Method to sample from a homogeneous Poisson process. Parameters:

        - lambdaHom: the homogeneous intensity value of the process (lambda)
        """
        self.lambdaHom = lambdaHom

        # Sample from a Poisson distribution to get the number of points
        self.N = ss.poisson(mu=self.lambdaHom * self.area * (enlarge ** 2)).rvs(1)

        # Compute new grid coordinates due to enlargement - this is for the case
        # of parent-child processes that inherit from this class
        midX = (self.minX + self.maxX)/2; midY = (self.minY + self.maxY)/2
        minXEnlarge = midX - self.lenX/2 * (1 + (enlarge - 1)/2)
        minYEnlarge = midY - self.lenY/2 * (1 + (enlarge - 1)/2)

        # Uniformly distribute the points on the window
        xHom = self.lenX * enlarge * ss.uniform(0,1).rvs(self.N) + minXEnlarge
        yHom = self.lenY * enlarge * ss.uniform(0,1).rvs(self.N) + minYEnlarge

        return np.array([xHom, yHom]).reshape((len(xHom), 2))

    def sample_and_plot(self, lambdaHom, file_name='', save=False):
        """
        Function to sample a homogeneous PP and to plot the result.
        """
        self.homPattern = self.simSPP(lambdaHom)
        self.plot(self.homPattern[:,0], self.homPattern[:,1], save, file_name)

    @staticmethod
    def plot(x, y, save, file_name):
        plt.scatter(x, y, edgecolor='b', alpha=0.5)
        plt.xlabel("x"); plt.ylabel("y")
        if save:
            if file_name == None:
                raise ValueError("Please give a file name.")
            plt.savefig(file_name)



class SPP_InhomPoisson(SPP_HomPoisson):
    """
    A class to simulate Poisson process (homogeneous and inhomogeneous) on 
    a rectangular window. Parameters:

    - minX, maxX, minY, maxY: end coordinates of the simulation window
    """
    def __init__(self, minX=0, maxX=1, minY=0, maxY=1):
        super().__init__(minX, maxX, minY, maxY)

    def simSPP(self, lambdaInhom) -> np.array:
        """
        Method to sample from a homogeneous Poisson process. In this
        method we sample using random thinning. Parameters:

        - lambdaInhom: the intensity function of the process. This must
                        be a two-parameter function outputting a single,
                        real number.
        """
        # Functions needed to compute the upper bound of the intensity
        @staticmethod
        def minimise_fun(z):
            return -lambdaInhom(z[0], z[1])

        @staticmethod
        def upper_bound(minimise_fun, minX, maxX, minY, maxY):
            init_xy = [(minX + maxX)/2, (minY + maxY)/2]
            opt = minimize(minimise_fun, init_xy, bounds = ((minX, maxX), (minY, maxY)))
            M = -opt.fun
            return M

        M = upper_bound(
            minimise_fun,
            self.minX,
            self.maxX,
            self.minY,
            self.maxY
        )
        
        # Simulate homogeneous Poisson process
        homPattern = super().simSPP(M)

        # Compute retention probability and Boolean for each point
        retainProb = (
            lambdaInhom(homPattern[:,0], homPattern[:,1]) / M
        )
        retainBool = (
            ss.uniform(0,1).rvs(len(homPattern)) < retainProb
        )
        
        # Form resulting inhomogeneous process
        return homPattern[retainBool,:]

    def sample_and_plot(self, lambdaInhom, file_name='', save=False):
        """
        Function to sample and plot the result.
        """
        self.inhomPattern = self.simSPP(lambdaInhom)
        self.plot(self.inhomPattern[:,0], self.inhomPattern[:,1],
                  save, file_name)


class SPP_Thomas(SPP_HomPoisson):
    """
    A class to simulate a Thomas point process. Parameters:

    - minX, maxX, minY, maxY: end coordinates of the simulation window
    """
    def __init__(self, minX=0, maxX=1, minY=0, maxY=1):
        super().__init__(minX, maxX, minY, maxY)

    def simSPP(self, rho, K, sigma, cov, enlarge) -> np.array:
        """
        Simulate a realisation of a Thomas point process. We can do this
        by simulating a homogeneous poisson process with intensity rho,
        and then simulating the number of children of each cluster as
        coming from Poisson(K) and simulating their locations using
        a Gaussian kernel with covaraince sigma^2I. Parameters:

        - rho: parent intensity.
        - K: intensity of Poisson distribution for number of offspring.
        - sigma: sd along diagonal of covariance matrix.
        - cov: the covariance matrix for the normal distribution. 
        - enlarge: scale factor to give grid to sim on to correct
                   edge effects.
        """
        # Simulate the parents (this is done on a larger grid than given
        # to account for edge effects)
        homPattern = super().simSPP(rho, enlarge)
        self.homPattern = homPattern

        # Iterate over each parent and simulate offspring using a 2d
        # Gaussian kernel
        numParents = len(homPattern)
        offspring = []
        for i in range(numParents):
            numOffspring = np.random.poisson(lam=K, size=1)[0]
            sampOffspring = np.random.multivariate_normal(
                mean=homPattern[i,:],
                cov=sigma**2 * cov,
                size=numOffspring
            )
            offspring.append(sampOffspring)

        # Concatenate lists into one array
        offspring = np.concatenate(offspring)

        # Drop rows where samples are outside of the domain
        inside = (
            (offspring[:,0] > self.minX) &
            (offspring[:,0] < self.maxX) &
            (offspring[:,1] > self.minY) &
            (offspring[:,1] < self.maxY)
        )

        return offspring[inside,:]

    def sample_and_plot(self, rho, K, sigma, cov, enlarge, file_name=None,
                        save=False):
        """
        Function to sample a homogeneous PP and to plot the result.
        """
        thomasSPP = self.simSPP(rho, K, sigma, cov, enlarge)
        self.plot(thomasSPP[:,0], thomasSPP[:,1], save, file_name)


class SPP_LGCP(SPP):
    """
    A class to simulate an LGCP on a rectangular window. 
    Parameters:
    - minX, maxX, minY, maxY: end coordinates of the simulation window.
    - step_size: lattice point spacing
    """
    def __init__(self, step_size, minX=0, maxX=1, minY=0, maxY=1):
        self.minX = minX; self.maxX = maxX 
        self.minY = minY; self.maxY = maxY
        self.step_size = step_size
 
        # Compute parameters of the simulation window
        self.lenX = maxX - minX; self.lenY = maxY - minY
        self.area = self.lenX * self.lenY

    def _sim_GRF(self, kernel, kernel_params, mean):
        """
        Method to sample the underlying GRF.
        Parameters:
            - kernel: a covmodel from GSTools.
            - kernel_params: a dictionary of the relevant parameters
                             for the inputted kernel.
        """
        model = kernel(**kernel_params)
        srf = SRF(model, mean=mean)
        x = y = np.arange(self.minX + self.step_size/2, self.maxX, self.step_size)
        self.field = srf.structured([x,y])
        self.srf = srf

    def simSPP(self, kernel, kernel_params, mean):
        """
        Method to sample the LGCP.
        Parameters:
            - kernel: a covmodel from GSTools.
            - kernel_params: a dictionary of the relevant parameters
                             for the inputted kernel.
            - mean: the baseline intensity
        """
        cell_area = self.step_size ** 2
        
        # Simulate the underlying GRF
        self._sim_GRF(kernel, kernel_params, mean)

        # Create the lattice to simulate the points on
        x = y = np.arange(self.minX + self.step_size/2, self.maxX, self.step_size)

        # List to store the cell_PPs
        full_PP = []

        # Iterate over the cells
        for i, x_lat in enumerate(x):
            for j, y_lat in enumerate(y):
                # Sample from a Poisson distribution to get the number of points
                N_cell = ss.poisson(mu=np.exp(self.field[i,j]) * cell_area).rvs(1)

                if N_cell != 0:
                    # Uniformly distribute the points on the window
                    xHom = np.random.uniform(x_lat - self.step_size/2,
                                    x_lat + self.step_size/2, N_cell) 
                    yHom = np.random.uniform(y_lat - self.step_size/2,
                                    y_lat + self.step_size/2, N_cell) 

                    cell_PP = np.array([xHom, yHom]).reshape((len(xHom), 2))
                    full_PP += [cell_PP]

        
        if len(full_PP) != 0:
            self.LGCP_sampled = np.vstack(full_PP)
            return self.LGCP_sampled

    def plot(self):
        """
        A method to plot the GRF with the sampled point pattern overlayed.
        """
        # Define a grid for the field
        x = np.linspace(0, self.maxX, 100) 
        y = np.linspace(0, self.maxY, 100)
        X, Y = np.meshgrid(x, y)

        # Reshape the field for plotting
        grf_surface = self.field.reshape(X.shape).T

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, grf_surface, levels=20, cmap="viridis")
        plt.colorbar(label="GRF Value")
        plt.scatter(self.LGCP_sampled[:,0], self.LGCP_sampled[:,1], 
                    color="red", label="Scatter Points")
        plt.title("GRF Surface with PP overlayed")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.show()

    def sample_and_plot(self):
        pass
