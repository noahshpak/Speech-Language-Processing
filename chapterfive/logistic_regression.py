import numpy as np


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

    def decision_function(self, X):
        """
        Predict confidence scores for samples
        :param X: array_like, shape (n_samples, n_features)
        :return: array, shape=(n_samples,)
        """
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
        pass

    def get_params(self, deep=True):
        """
        Get Parameters for this estimator
        :param deep: if True will return nested params from other estimators
        :return: params: mapping of string to any
        """
        pass

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Perform one epoch of SGD on given samples
        :param X:  array-like, shape (n_samples, n_features)
            subset of training data
        :param y: array, shape (n_samples)
            subset of target values
        :param classes: array, shape (n_classes)
            Classes across all calls to partial_fit.
            Can be obtained by via np.unique(y_all),
            where y_all is the target vector of the entire dataset.
            This argument is required for the first call to partial_fit and can be omitted in the subsequent calls.
            Note that y doesn’t need to contain all labels in classes.
        :param sample_weight: array-like, shape (n_samples), optional
            Weights applied to individual examples. If not supplied, uniform weights assumed.
        :return: self
        """
        pass

    def predict(self, X):
        """
        predict class labels for samples in X
        :param X: array-like, shape (n_samples, n_features)
            Samples
        :return: C: array, shape [n_samples]
            Predicted class label per sample
        """
        pass

    def predict_log_proba(self, X):
        """
        Log of probability estimates

        Only available for Log Loss and modified Huber loss
        :param X: array-like, shape (n_samples, n_features)
        :return: T: array-like, shape (n_samples, n_classes)
        Returns the log-probability of the sample for each class in the model,
        where classes are ordered as they are in self.classes_
        """
        pass

    def predict_proba(self):
        """
        Only available for log loss

        :param X: array-like, shape (n_samples, n_features)
        :return: T: array-like, shape (n_samples, n_classes)
        Returns the probability of the sample for each class in the model,
        where classes are ordered as they are in self.classes_
        pass
        """
    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels
        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples)
        :param sample_weight: array-like, shape [n_samples]
        :return: score: float
            Mean accuracy of self.predict(X) wrt y.
        """
        pass
