import torch
import pytest

from chapterfive.logistic_regression import LogisticRegressionClassifier

def test_random_training():
    num_samples = 256
    num_features = 5
    x = torch.randn(num_samples, num_features).float()
    y = torch.randint(2, (num_samples, 1)).float()

    LR = LogisticRegressionClassifier()
    LR.fit(x, y, batch_size=32)

    assert LR.score(x, y) >= 0.5

