import numpy as np
from utils.distance import euclidean_dist

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

    DBSCAN is a clustering algorithm that groups together data points that are close to each other based on a distance measure.
    It marks points as core points if they have at least `min_pts` neighbors within a radius of `eps`.
    Points that are neighbors of core points are marked as border points.
    Points that are not core points or border points are marked as noise.

    Parameters:
        eps (float): The maximum distance between two points to be considered neighbors.
        min_pts (int): The minimum number of neighbors a point must have to be considered a core point.

    Attributes:
        eps (float): The maximum distance between two points to be considered neighbors.
        min_pts (int): The minimum number of neighbors a point must have to be considered a core point.
        clusters (list[list]): A list of clusters, where each cluster is a list of indices of the points in the cluster.
        labels (numpy.ndarray): An array of labels, where each label corresponds to the cluster of the corresponding point.

    Methods:
        fit(X): Fits the DBSCAN algorithm to the data `X`.
        _get_neighbors(p_idx, X): Gets the neighbors of a point `p_idx` in the data `X`.
        _expand_cluster(idx, neighbors, X): Expands a cluster from a core point `idx` and its neighbors `neighbors` in the data `X`.
        _get_labels(X): Assigns labels to points in the data `X` based on their cluster memberships.
    """

    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts

    def _get_neighbors(self, p_idx, X):
        """Gets the neighbors of a point `p_idx` in the data `X`.

        Parameters:
            p_idx (int): The index of the point to get the neighbors of.
            X (numpy.ndarray): The data.

        Returns:
            list[int]: The indices of the neighbors of `p_idx`.
        """

        if p_idx in self._computed_neighbors:
            return self._computed_neighbors[p_idx]

        neighbors = []
        for idx, p in enumerate(X):
            if p_idx == idx:
                continue
            if euclidean_dist(X[idx], X[p_idx]) < self.eps:
                neighbors.append(idx)

        self._computed_neighbors[p_idx] = neighbors
        return neighbors

    def _expand_cluster(self, idx, neighbors, X):
        """Expands a cluster from a core point `idx` and its neighbors `neighbors` in the data `X`.

        Parameters:
            idx (int): The index of the core point.
            neighbors (list[int]): The indices of the neighbors of the core point.
            X (numpy.ndarray): The data.

        Returns:
            list[int]: The indices of the points in the cluster.
        """

        cluster = [idx]
        for neigh_idx in neighbors:
            if self.marked[neigh_idx]:
                continue
            self.marked[neigh_idx] = True
            cur_neighbors = self._get_neighbors(neigh_idx, X)
            if len(cur_neighbors) >= self.min_pts:
                # Expand further if core point
                cluster += self._expand_cluster(neigh_idx, cur_neighbors, X)
            else:
                # Append if border point
                cluster.append(neigh_idx)

        return cluster

    def _get_labels(self, X):
        """Assigns labels to points in the data `X` based on their cluster memberships"""

        labels = np.zeros(len(X))
        for label, cluster in enumerate(self.clusters):
            for p in cluster:
                labels[p] = label
        return labels
        
    def fit(self, X):
        """Fits the DBSCAN algorithm to the data `X`.

        Parameters:
            X (numpy.ndarray): The feature matrix to fit the DBSCAN algorithm to.

        Returns:
            tuple[list[list], numpy.ndarray]: A tuple containing the clusters and labels of the data.
        """

        self.marked = np.full(len(X), False)
        self._computed_neighbors = {}
        self.clusters = []
        for idx, p in enumerate(X):
            if self.marked[idx]:
                continue

            cur_neighbors = self._get_neighbors(idx, X)

            # If core point
            if len(cur_neighbors) >= self.min_pts:
                self.marked[idx] = True
                new_cluster = self._expand_cluster(idx, cur_neighbors, X)
                self.clusters.append(new_cluster)

        self.labels = self._get_labels(X)
        return self.clusters, self.labels
        