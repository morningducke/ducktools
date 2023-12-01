import numpy as np
from utils.distance import euclidean_dist

class KMeans:
    """K-means clustering algorithm
    K-means clustering is a unsupervised machine learning algorithm that groups data points into a specified number of clusters (k).
    The algorithm works by iteratively assigning data points to the nearest cluster centroid and then updating the cluster centroids
    until a stopping criterion is met.
    Parameters:
        k (int): The number of clusters to group the data into.
        max_iter (int): The maximum number of iterations to run the k-means algorithm.
    Attributes:
        k (int): The number of clusters to group the data into.
        max_iter (int): The maximum number of iterations to run the k-means algorithm.
        rng (numpy.random.default_rng): A random number generator used to initialize the cluster centroids.
        history (list[list]): A list of tuples containing the total distortion and centroids for each iteration of the k-means algorithm.
    Methods:
        fit(X): Fits the k-means algorithm to the data `X`.
        euclidean_dist(v1, v2): Calculates the Euclidean distance between two vectors `v1` and `v2`.
        predict(X): Predicts the cluster labels for the data `X`.
    """
        
    def __init__(self, k, max_iter=200):
        self.k = k
        self.max_iter = max_iter
        self.rng = np.random.default_rng()
        self.history = []

    
    def fit(self, X):
        """Fits the k-means algorithm to the data `X`.

        Parameters:
            X (numpy.ndarray): The feature matrix to fit the k-means algorithm to.

        Returns:
            self: The k-means model.
        """
        self.centroids = self.rng.choice(X, self.k, replace=False)
        self.labels = np.zeros(len(X))
        for i in range(self.max_iter):
            self.means = np.zeros((self.k, X.shape[1]))
            self.label_count = np.zeros(self.k)
            total_dist = 0
            # for each point calc distance to centroids then choose best
            for idx, p in enumerate(X):
                dist = euclidean_dist(self.centroids, p)
                best_centroid = np.argmin(dist)
                total_dist += dist[best_centroid]
                self.labels[idx] = best_centroid
                self.label_count[best_centroid] += 1
                self.means[best_centroid] += p
            # update centroids
            self.history.append([total_dist / len(X), self.centroids]) 
            self.centroids = self.means / self.label_count[:, None]
        return self

    def predict(self, X):
        """Predicts the cluster labels for the data `X`.

        Parameters:
            X (numpy.ndarray): The data to predict the cluster labels for.

        Returns:
            numpy.ndarray: The cluster labels for the data `X`.
        """
        labels = []
        for p in X:
            dist = euclidean_dist(self.centroids, p)
            labels.append(np.argmin(dist))
        return labels