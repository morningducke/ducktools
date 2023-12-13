import numpy as np

class LinearSVM:
    """Support vector machine (SVM) is a discriminative algorithm mostly used for classification.
    The crux of SVM is building a hyperplane by maximizing the distance between the closest points (support vectors) and the hyperplane.
    This implementation only supports binary classiification with labels 0 or 1.
    
    Parameters:
        lr: learning rate
        iters: number of epochs
        l2_reg: lambda parameter for the L2 regularization

    Attributes:
        lr: learning rate
        iters: number of epochs
        l2_reg: lambda parameter for the L2 regularization

    Methods:
        fit(X, y): fits the hyperplane to data X
        predict(X): predicts labels for X
    """
    
    def __init__(self, lr=1e-3, iters=25, l2_reg=1e-5):
        self.lr = lr
        self.iters = iters
        self.l2_reg = l2_reg

    def _add_bias_term(self, X):
        """Adds a column of ones corresponding to the bias term"""
        
        return np.c_[X, np.ones(len(X))]
        
    def _transform_labels(self, y):
        """Replaces 0 labels with -1 to accomodate the SVM algorithm"""
        
        return np.where(y == 0, -1, 1)

    def fit(self, X, y):
        """Fits a hyperplane to data

        Parameters:
            X: numpy array of data points
            y: numpy array of labels (either 0 or 1) 
        """
        
        X_transformed = self._add_bias_term(X)
        self._w = np.random.rand(X.shape[1] + 1)
        y_transformed = self._transform_labels(y)

        for _ in range(self.iters):
            self._step(X_transformed, y_transformed)

    def _step(self, X, y):
        """Calculates and applies gradients for current weights"""
        
        margins = y * X.dot(self._w.T)
        dw = np.where(margins[:, None] >= 1, 2 * self.l2_reg * self._w, 2 * self.l2_reg * self._w - X * y[:, None]).sum(axis=0)
        self._w -= self.lr * dw
        
    def predict(self, X):
        """Predicts labels for given data

        Parameters:
            X: numpy array of data points

        Returns:
            A numpy array of predicted labels
        """
        X_transformed = self._add_bias_term(X)
        preds = np.sign(X_transformed.dot(self._w.T))
        return np.where(preds <= -1, 0, 1)