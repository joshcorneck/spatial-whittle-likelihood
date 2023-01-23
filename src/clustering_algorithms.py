import numpy as np
import pandas as pd
import sklearn.cluster as sc
from sklearn.metrics import silhouette_score

from spatial_pp import SPP_Thomas

class KMeansEstimation:
    """
    A class to implement my Kmeans estimation procedure. A point pattern from
    a Thomas process is inputted. A number of clusters is estimated and then
    the pattern is clustered. THe number of clusters is an estimate for rho, 
    the number of points in each cluster is used to estimate K and the patterns
    in each cluster is used to estimate sigma (using their centroid as an estimate
    of their mean).
    """
    def __init__(self, spp):
        self.spp = spp

    def selectNumberOfClusters(self):

        range_num_clusters = np.arange(10, 40 + 1, 1)
        silhouette_avg = []

        for num_clusters in range_num_clusters:
            kmeans = sc.KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(self.spp)
            cluster_labels = kmeans.labels_

            # Silhouette score
            silhouette_avg.append(silhouette_score(self.spp, cluster_labels))

        return (np.argmax(silhouette_avg) + 1)

    def clusterPoints(self, K):
        """
        Cluster all the points into K different clusters. Parameters:

        - K: number of clusters.
        """
        self.K = K
        km = sc.KMeans(n_clusters=K, n_init=10)
        km.fit(self.spp)
        clusters = km.labels_
        self.spp_clusters = (
            np.append(self.spp, clusters.reshape(len(clusters),1), axis=1))
        self.cluster_centres = km.cluster_centers_

    def estimateRho(self):
        """
        For now, simply use the number of clusters as our estimate of rho.
        """

        rho_mle = self.K

        return rho_mle
    
    def estimateK(self):
        """
        Use the MLE of a Poisson process to estimate K.
        """
        K_mle = (pd.DataFrame({'x': self.spp_clusters[:,0], 
                      'y': self.spp_clusters[:,1], 
                      'cluster': self.spp_clusters[:,2]}).
                      groupby(['cluster']).size().mean())

        return K_mle
    
    def estimateSigma(self):
        """
        We assume independence of x and y and isotropy, so we can simply 
        estimate sigma using the mle of a univariate normal distribution.
        """
        # Join cluster centres onto pp_cluster and subtract to shift
        # all points to be centred about the origin
        df_pp_cluster = (pd.DataFrame({'x': self.spp_clusters[:,0], 
                      'y': self.spp_clusters[:,1], 
                      'cluster': self.spp_clusters[:,2]}))
        df_cluster_centres = (pd.DataFrame(
            {'cluster': np.arange(0, self.K, 1),
             'centre_x': self.cluster_centres[:,0],
             'centre_y': self.cluster_centres[:,1]}))
        df_pp_cluster_centres = (
            df_pp_cluster.merge(df_cluster_centres,
                                on='cluster',
                                how='left')
        )
        pp_centred = (np.array(
            [(df_pp_cluster_centres.x - df_pp_cluster_centres.centre_x),
              df_pp_cluster_centres.y - df_pp_cluster_centres.centre_y]).T
             )

        sigma_mle = np.sqrt(
            np.mean(
                pp_centred ** 2
            )
        )

        return sigma_mle

    def computeEstimators(self, K):

        self.clusterPoints(K)
        rho_est = self.estimateRho()
        K_est = self.estimateK()
        sigma_est = self.estimateSigma()

        return [rho_est, K_est, sigma_est]





        



        
