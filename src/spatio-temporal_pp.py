import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import minimize

class STPP_Poisson:
    """
    A class to simulate a spatial-temporal Poisson process.
    """
    def __init__(self, minX=0, maxX=1, minY=0, maxY=1,
                minT=0, maxT=10):
        self.minX = minX; self.maxX = maxX 
        self.minY = minY; self.maxY = maxY
        self.minT = minT; self.maxT = maxT

        # Compute parameters of the simulation window
        self.lenX = maxX - minX; self.lenY = maxY - minY
        self.lenT = self.maxT - self.minT
        self.vol = self.lenX * self.lenY * self.lenT

    def homSTPP(self, lambdaHom):
        """
        Method to sample from a homogeneous Poisson process. Parameters:

        - lambdaHom: the homogeneous intensity value of the process (lambda)
        """
        self.lambdaHom = lambdaHom

        # Sample from a Poisson distribution to get the number of points
        self.N = ss.poisson(mu=self.lambdaHom * self.vol).rvs(1)

        # Uniformly distribute the points on the window
        xHom = self.lenX * ss.uniform(0,1).rvs(self.N) + self.minX
        yHom = self.lenY * ss.uniform(0,1).rvs(self.N) + self.minY
        tHom = self.lenT * ss.uniform(0,1).rvs(self.N) + self.minT
        self.homPattern = np.array([xHom, yHom, tHom])

    def inhomPP_random_thinning(self, lambdaInhom):
        """
        Method to sample from a homogeneous Poisson process. In this
        method we sample using random thinning. Parameters:

        - lambdaInhom: the intensity function of the process. This must
                        be a three-parameter function outputting a single,
                        real number. Inputs are R^2 spatial coordinates
                        and time.
        """
        # Functions needed to compute the upper bound of the intensity
        def minimise_fun(z):
            return -lambdaInhom(z[0], z[1], z[2])

        def upper_bound(minimise_fun, minX, maxX, minY, maxY, minT, maxT):
            init_xyt = [(minX + maxX)/2, (minY + maxY)/2, (minT + maxT)/2]
            opt = minimize(
                minimise_fun, 
                init_xyt, 
                bounds = ((minX, maxX), (minY, maxY), (minT, maxT)))
            M = -opt.fun
            return M

        M = upper_bound(
            minimise_fun, 
            self.minX,
            self.maxX,
            self.minY,
            self.maxY,
            self.minT,
            self.maxT
        )
        
        # Simulate homogeneous process
        self.homSTPP(M)

        # Compute retention probability and Boolean for each point
        retainProb = (
            lambdaInhom(
                self.homPattern[0,:], 
                self.homPattern[1,:],
                self.homPattern[2,:]) / M
        )
        retainBool = (
            ss.uniform(0,1).rvs(len(self.homPattern[0,:])) < retainProb
        )
        
        # Form resulting inhomogeneous process
        self.inhomPattern = (
            [self.homPattern[0, retainBool], 
            self.homPattern[1, retainBool],
            self.homPattern[2, retainBool]]
        )

    def sample_and_plot_hom(self, lambdaHom):
        """
        Function to sample a homogeneous STPP and to plot the result.
        """
        self.homSTPP(lambdaHom)
        self.plot(
            self.homPattern[0,:], 
            self.homPattern[1,:],
            self.homPattern[3,:])

    def sample_and_plot_inhom(self, lambdaInhom):
        """
        Function to sample an inhomogeneous STPP and to plot the result.
        """
        self.inhomPP_random_thinning(lambdaInhom)
        self.plot(
            self.inhomPattern[0][:], 
            self.inhomPattern[1][:],
            self.inhomPattern[2][:])

    @staticmethod
    def plot(x, y, t):
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, t, c=np.round(t))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
        plt.show()


class STPP_LGCP:
    """
    A class to simulate a spatial-temporal log-Gaussian Cox processes.
    """
    def __init__(self, minX=0, maxX=1, minY=0, maxY=1,
                minT=0, maxT=10):
        self.minX = minX; self.maxX = maxX 
        self.minY = minY; self.maxY = maxY
        self.minT = minT; self.maxT = maxT

        # Compute parameters of the simulation window
        self.lenX = maxX - minX; self.lenY = maxY - minY
        self.lenT = self.maxT - self.minT
        self.vol = self.lenX * self.lenY * self.lenT

    def homSTPP(self, lambdaHom):
        """
        Method to sample from a homogeneous Poisson process. Parameters:

        - lambdaHom: the homogeneous intensity value of the process (lambda)
        """
        self.lambdaHom = lambdaHom

        # Sample from a Poisson distribution to get the number of points
        self.N = ss.poisson(mu=self.lambdaHom * self.vol).rvs(1)

        # Uniformly distribute the points on the window
        xHom = self.lenX * ss.uniform(0,1).rvs(self.N) + self.minX
        yHom = self.lenY * ss.uniform(0,1).rvs(self.N) + self.minY
        tHom = self.lenT * ss.uniform(0,1).rvs(self.N) + self.minT
        self.homPattern = np.array([xHom, yHom, tHom])

    def simulate_GRF(self):
        """
        Method to simulate a Gaussian random field. Flexibility is given
        for the choice of covariance function and subsequent parameter
        choices. Anisotropy in space and time is also allowed.
        """
        # Spatial and temporal grid points
        x = np.linspace(self.minX, self.maxX, 100)
        y = np.linspace(self.minY, self.maxY, 100)
        t = np.linspace(self.minT, self.maxT, 10)

        # Total spatio-temporal dimension
        st_dim = 2 + 1

        # An exponential variogram 
        model = gs.Exponential(dim=st_dim, var=1.0, len_scale=0.1)

        # Create a spatial random field instance
        srf = gs.SRF(model)
        
        # The Gaussian random field
        pos, time = [x, y], [t]
        self.field = np.exp(srf.structured(pos + time))

    def produce_LGCP(self, lambdaInhom):
        """
        Method to sample from a homogeneous Poisson process. In this
        method we sample using random thinning. Parameters:

        - lambdaInhom: the intensity function of the process. This must
                        be a three-parameter function outputting a single,
                        real number. Inputs are R^2 spatial coordinates
                        and time.
        """
        # Produce a realisation of the GRF
        self.simulate_GRF()

        # Upper bound of the intensity
        M = np.max(self.field)
        
        # Simulate homogeneous process
        self.homSTPP(M)

        # Compute retention probability and Boolean for each point
        retainProb = (
            lambdaInhom(
                self.homPattern[0,:], 
                self.homPattern[1,:],
                self.homPattern[2,:]) / M
        )
        retainBool = (
            ss.uniform(0,1).rvs(len(self.homPattern[0,:])) < retainProb
        )
        
        # Form resulting inhomogeneous process
        self.inhomPattern = (
            [self.homPattern[0, retainBool], 
            self.homPattern[1, retainBool],
            self.homPattern[2, retainBool]]
        )

    def sample_and_plot_hom(self, lambdaHom):
        """
        Function to sample a homogeneous STPP and to plot the result.
        """
        self.homSTPP(lambdaHom)
        self.plot(
            self.homPattern[0,:], 
            self.homPattern[1,:],
            self.homPattern[3,:])

    def sample_and_plot_inhom(self, lambdaInhom):
        """
        Function to sample an inhomogeneous STPP and to plot the result.
        """
        self.inhomPP_random_thinning(lambdaInhom)
        self.plot(
            self.inhomPattern[0][:], 
            self.inhomPattern[1][:],
            self.inhomPattern[2][:])

    @staticmethod
    def plot(x, y, t):
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, t, c=np.round(t))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
        plt.show()


class HawkesIntensity:
    """
    A class that defines a blueprint for the spatio-temporal Hawkes
    intensity function. Parameters:
    
    - nu: background intensity
    - trigger: trigger function
    """
    def __init__(self, mu, trigger):
        self.nu = mu
        self.trigger = trigger

    def compute(self, s, t, his_s=None, his_t=None):
        """
        Compute the value of the intensity at (s,t).
        """
        if (his_s is None) | (his_t is None):
            ints = self.mu
        else:
            ints = self.mu + self.trigger.g(s, t, his_s, his_t)
        return ints

class Trigger_ExpGauss:
    """
    Trigger function compose of a product of an exponential function in
    time and a Gaussian in space.
    """
    def __init__(self, omega, sigx, sigy):
        self.omega = omega
        self.sigx = sigx
        self.sigy = sigy

    def int_over_support(self, x1=0, x2=1, y1=0, 
                         y2=1, T=10):
        """
        Method to compute m - the integral over the support
        of the trigger function.
        """
        m = ((1 - np.exp(-self.omega * T)) * 
            (ss.norm.cdf(x2/self.sigx) - ss.norm.cdf(x1/self.sigx)) * 
            (ss.norm.cdf(y2/self.sigy) - ss.norm.cdf(y1/self.sigy)))

        return m

    def maximum(self):
        return 10 * self.omega / (2 * np.pi * self.sigx ** 2 * self.sigy ** 2)

    def g(self, x, y, t):
        """
        Evaluate the trigger function at (t,x,y)
        """
        g = 10 * (self.omega/(2 * np.pi * self.sigx ** 2 * self.sigy ** 2) *
            np.exp(-self.omega * t) * 
            np.exp(-x ** 2/(2 * self.sigx ** 2)) * 
            np.exp(-y ** 2/(2 * self.sigy ** 2)))

        return g

            




class STPP_Hawkes:
    """
    A class to simulate realisations of a spatio-temporal Hawkes process.
    Parameters:

    - intensity: an instance of the HawkesIntensity class.
    - minX, maxX, minY, maxY, minT, maxT: spatial and temporal windows.
    """
    def __init__(self, intensity, minX=0, maxX=1, minY=0, maxY=1,
                 minT=0, maxT=10):
        self.minX = minX; self.maxX = maxX 
        self.minY = minY; self.maxY = maxY
        self.minT = minT; self.maxT = maxT
        self.intensity = intensity

        # Compute parameters of the simulation window
        self.lenX = maxX - minX; self.lenY = maxY - minY
        self.lenT = self.maxT - self.minT
        self.vol = self.lenX * self.lenY * self.lenT

    def homSTPP(self):
        """
        Method to sample from a homogeneous Poisson process. Parameters:

        - lambdaHom: the homogeneous intensity value of the process (lambda)
        """
        # 10 is current mu value
        self.M = 10 + self.intensity.maximum() 

        # Sample from a Poisson distribution to get the number of points
        self.N = ss.poisson(mu=self.M * self.vol).rvs(1)[0]

        # Uniformly distribute the points on the window
        xHom = self.lenX * ss.uniform(0,1).rvs(self.N) + self.minX
        yHom = self.lenY * ss.uniform(0,1).rvs(self.N) + self.minY
        tHom = self.lenT * ss.uniform(0,1).rvs(self.N) + self.minT

        # Sort by time index
        homPattern = np.array([xHom, yHom, tHom])
        homPattern = homPattern[:, homPattern[2,:].argsort()]
        self.homPattern = homPattern

    def thin_homSTPP(self):
        """
        Thin a sampled homogeneous Poisson STPP according to the supplied
        intensity function.
        """
        self.homSTPP()
        self.hawkes = self.homPattern
        keep = [0]
        for i in range(self.N):
            if i == 0:
                pass
            else:
                g_vals = np.zeros(i)
                for j in keep:
                    g_vals[j] = self.intensity.g(
                        self.hawkes[0,i] - self.hawkes[0,j],
                        self.hawkes[1,i] - self.hawkes[1,j],
                        self.hawkes[2,i] - self.hawkes[2,j]
                    )
                # 10 is current mu value
                lambda_val = 10 + np.sum(g_vals)
                retainProb = lambda_val / self.M
                # print(retainProb)
                retainBool = (np.random.uniform(0, 1, 1) < retainProb)
                if retainBool:
                    keep.append(i)
        self.hawkes = self.hawkes[:, keep]
                    

def plot(x, y, t):
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, t, c=np.round(t))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
        plt.show()

