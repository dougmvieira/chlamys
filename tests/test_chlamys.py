from itertools import product

import numpy as np

import chlamys as ch


def test_interp_levels():
    np.random.seed(0)
    d = 7
    n = 20

    points = np.array(list(product(range(2), repeat=d)))
    values = ch.test_function(*np.transpose(points))
    interpolands = np.random.rand(d, n)

    ys_bench = np.mean(values)
    ys_hat = ch.interp_levels(points, values, interpolands)
    ys = ch.test_function(*interpolands)

    assert np.max(np.abs(ys - ys_hat)) < np.max(np.abs(ys - ys_bench))


def test_interp_1st_order():
    np.random.seed(0)
    d = 3
    n = 20

    points = np.array(list(product(range(2), repeat=d)))
    values = ch.test_function(*np.transpose(points))
    grads = np.stack(ch.test_gradient(*np.transpose(points)), axis=-1)
    interpolands = np.random.rand(d, n)

    ys_bench = ch.interp_levels(points, values, interpolands)
    ys_hat = ch.interp_1st_order(points, values, grads, interpolands)
    ys = ch.test_function(*interpolands)

    assert np.max(np.abs(ys - ys_hat)) < np.max(np.abs(ys - ys_bench))
