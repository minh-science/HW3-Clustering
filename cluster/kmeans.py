import numpy as np
from scipy.spatial.distance import cdist

import random as rd

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k_clusters = k
        self.tolerance = tol
        self.maximum_iterations = max_iter
        self.centroids = None
        self.error = None

        if k < 1:
            raise ValueError("Invalid K value")
        

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        n, m = mat.shape
        if n < self.k_clusters:
            raise ValueError("Invalid k value, k should be less than number of observations")
        # print("n", n)

        self.centroids = mat[np.random.choice(n, self.k_clusters, replace=False)]
            

        for _ in range(self.maximum_iterations):
            # Assign each observation to the nearest centroid
            labels = np.argmin(cdist(mat, self.centroids), axis=1)

            # Update centroids
            new_centroids = np.array([mat[labels == i].mean(axis=0) for i in range(self.k_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                # print("loooooop")
                return
        
            self.centroids = new_centroids

        # Error calculation, sum of squares 
        self.error = np.sum(np.min(cdist(mat, self.centroids), axis=1)**2)




    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Input matrix has different number of features than the fitted data.")
        
        predicted_labels = np.argmin(cdist(mat, self.centroids), axis=1)

        return predicted_labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
