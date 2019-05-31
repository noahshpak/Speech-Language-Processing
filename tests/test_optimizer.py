import pytest
import numpy as np
from chapterfive import optimizer
from chapterfive import activations


def test_optimizer():
    x = np.array([3, 2])
    y = np.array([1])
    b = 0
    w = np.array([0, 0])
    lr = 0.1

    prediction = activations.sigmoid(np.dot(w, x) + b)
    gradient_w = np.dot((prediction - y), x)
    gradient_b = prediction - y
    gradient_w_b = np.concatenate([gradient_w, gradient_b])
    opt = optimizer.SGD(max_iter=1)
    assert opt.fit(x, y)


