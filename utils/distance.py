import numpy as np

def euclidean_dist(self, v1, v2):
        """
        Calculates the Euclidean distance between two points `v1` and `v2`.

        Parameters:
            v1 (numpy.ndarray): The first point.
            v2 (numpy.ndarray): The second point.

        Returns:
            float: The Euclidean distance between `v1` and `v2`.
        """
        return np.linalg.norm(v1 - v2)