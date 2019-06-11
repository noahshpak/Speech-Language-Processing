import math
import torch


class LogisticRegressionClassifier:
    """
    This implementation only covers Binary Classification
    Using the Scikit-Learn interface as inspiration
    (tip from Jeremy Howard)
    """
    def __init__(self, max_iter=100, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y, batch_size, coef_init=None, intercept_init=None):
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
        self.W = torch.randn(c, 1) / math.sqrt(c) if not coef_init else coef_init  # Glorot initialization
        self.W.requires_grad_()
        self.b = torch.zeros(1, requires_grad=True) if not intercept_init else intercept_init

        for i in range(self.max_iter):
            print(i)
            for xb, yb in self.batchify(X, y, batch_size):
                predictions = self.predict_proba(xb)  # forward pass
                loss = self.cross_entropy_loss(predictions, yb)
                print(loss)
                loss.backward()

                with torch.no_grad():
                    self.W -= self.W.grad * self.learning_rate
                    self.b -= self.b.grad * self.learning_rate
                    self.W.grad.zero_()
                    self.b.grad.zero_()

    @staticmethod
    def cross_entropy_loss(y_hat, y):
        return -(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat)).mean()

    @staticmethod
    def batchify(X, y, bsz):
        assert y.size()[0] == X.size()[0]
        num_samples = X.size()[0]
        for i in range(0, num_samples, bsz):
            yield X[i:i+bsz], y[i:i+bsz]

    def predict(self, X, decision_boundary=0.5):
        """
        predict class labels for samples in X
        :param decision_boundary: if pred proba > decision_boundary => True
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
        return (self.predict(X).float() == y).float().mean()
