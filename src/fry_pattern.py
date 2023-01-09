#%%
import numpy as np
import matplotlib.pyplot as plt

from spatial_pp import SPP_Thomas, SPP_HomPoisson

class FryPattern:

    def __init__(self, spp):
        """
        A class to comput the Fry pattern of an inputted point pattern
        realisation. Parameters:

        - spp: the 2D point pattern realisation. An N x 2 numpy array
        """
        self.spp = spp

    def compute_Fry_pattern(self):

        N = len(self.spp)
        fry_pattern = np.zeros((N*(N-1), 2))

        # Counter for indexing
        k = 0
        for i in range(N):
            for j in range(i+1, N):
                fry_pattern[k, :] = self.spp[i, :] - self.spp[j, :]
                k += 1

        fry_pattern[int(N*(N-1)/2):(N*(N-1)), :] = -(
            fry_pattern[0:int(N*(N-1)/2), :]
        )

        return fry_pattern

    def compute_and_plot(self):
        fry_pattern = self.compute_Fry_pattern()
        plt.scatter(fry_pattern[:,0], fry_pattern[:,1], alpha=0.2)
        plt.xlim([-0.3, 0.3]); plt.ylim([-0.3,0.3])

#%%
tom = SPP_Thomas()
spp = tom.simSPP(20, 5, 0.03, np.array([[1,0], [0, 1]]), 1.25)
hompp = SPP_HomPoisson()
spp = hompp.simSPP(lambdaHom=100)

fp = FryPattern(spp)
fp.compute_and_plot()

plt.figure(2)
plt.scatter(spp[:,0], spp[:,1])

# %%
