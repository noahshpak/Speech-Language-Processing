import math
import torch


class LogisticRegressionClassifier:
    """
    This implementation only covers Binary Classification
    Using the Scikit-Learn interface as inspiration
    (tip from Jeremy Howard)
    """
    def __init__(self, penalty='l2', max_iter=100, n_jobs=None, verbose=0, loss='log',
                 shuffle=True):
        self.penalty = penalty
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.loss = loss
        self.shuffle = shuffle

    @staticmethod
    def cross_entropy_loss(x, y):
        pass

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """
        Fit the logisitic regression with SGD
        :param X: array-like matrix with shape (n_samples, n_features)
            Training data
        :param y: numpy array shape (n_samples)
            Target values
        :param coef_init: array, shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization
        :param intercept_init: array, shape (n_classes, )
            The initial intercept to warm-start the optimization
        :param sample_weight: array-like shape (n_samples), optional
            Weights applied to individual examples. If not supplied, uniform weights assumed.
        :return: self
        """
        n, c = X.shape
        self.W = torch.randn(c, 1) / math.sqrt(c)  # Glorot initialization
        self.W.requires_grad_()
        self.b = torch.zeros(1, requires_grad=True)
        for xb in self.batchify(X):
            preds = torch.sigmoid(xb @ self.W + self.b)

    def batchify(self, X):
        pass

    def predict(self, X, decision_boundary=0.5):
        """
        predict class labels for samples in X
        :param X: array-like, shape (n_samples, n_features)
            Samples
        :return: C: array, shape [n_samples]
            Predicted class label per sample
        """
        proba = self.predict_proba(X)
        return proba > decision_boundary

    def predict_proba(self, X):
        """
        :param X: array-like, shape (n_samples, n_features)
        :return: the model's confidence in each prediction
        """
        return torch.sigmoid(X @ self.W + self.b)

    def score(self, X, y, sample_weight=None, decision_boundary=0.5):
        """
        Returns the mean accuracy on the given test data and labels
        :param decision_boundary: the confidence value that signifies pos classification (default: 0.5)
        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples)
        :param sample_weight: array-like, shape [n_samples]
        :return: score: float
            Mean accuracy of self.predict(X) wrt y.
        """
        assert y.size()[0] == X.size()[0]
        return torch.sum(self.predict(X) == y) / len(y)
