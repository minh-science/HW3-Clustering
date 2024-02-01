import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        n = len(X)
        silhouette_scores = np.zeros(n)

        for i in range(n):
            # Calculate the average distance of the ith observation to all other observations in the same cluster
            intra_cluster_distances = cdist(X[i:i+1, :], X[y == y[i]])
            a_i = np.mean(intra_cluster_distances)

            # Calculate the average distance of the ith observation to all observations in the nearest other cluster
            other_cluster_distances = cdist(X[i:i+1, :], X[y != y[i]])
            # print(other_cluster_distances)
            b_i = np.min(other_cluster_distances)

            # Calculate the silhouette score for the ith observation
            silhouette_scores[i] = (b_i - a_i) / max(b_i, a_i)

        return silhouette_scores
