import numpy as np

class SGD:
    """
    Stochastic Gradient Descent

    X:
    y:
    F:
    L:
    """
    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def fit(self, X, y, F, L, dL, epsilon=1e-3):
        """
        Compute the updates
        :param X: Training inputs of shape (n_samples, n_features)
        :param y: training labels of shape (n_samples,)
        :param F: a function parameterized, by `coef_`
        :param dL: the derivative of L with respect to w
        :param L: a loss function
        :param epsilon: learning rate
        :return: updated parameters
        """
        pass

