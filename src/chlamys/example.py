import numpy as np


def _dtanh_dx(x):
    return 1 - np.tanh(x)**2


def test_function(*xs):
    return sum(map(np.tanh, xs[::2])) + sum(map(np.cosh, xs[1::2]))


def test_gradient(*xs):
    grad = list(xs)
    grad[::2] = list(map(_dtanh_dx, xs[::2]))
    grad[1::2] = list(map(np.sinh, xs[1::2]))
    return tuple(grad)
