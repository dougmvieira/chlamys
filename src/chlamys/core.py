from itertools import chain

import numpy as np
import sympy as sy
from scipy import spatial


class Delaunay1D:
    def __init__(self, base):
        indices = np.argsort(base)

        self.simplices = np.stack([indices[:-1], indices[1:]])
        self.sorted_base = base[indices]
        self.convex_hull = indices[[0, -1]]

    def find_simplex(self, x):
        x = x.ravel()
        sorted_base = self.sorted_base

        found = np.searchsorted(sorted_base, x) - 1
        found[x == sorted_base[0]] = 0
        found[found == len(sorted_base)] = -1

        return found


def plane(xi, points, values):
    """Planar interpolation on a simplex

    Interpolation via an D-dimensional plane given D+1 points of a simplex.

    Parameters
    ----------
    xi : tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.
    points : ndarray of floats, shape (D+1, D)
        Data point coordinates.
    values : ndarray of floats, shape (D+1,)
        Data values.

    Returns
    -------
    ndarray
        Array of interpolated values.

    """

    alpha = np.linalg.solve(points[1:] - points[0], values[1:] - values[0])
    return (np.stack(xi, axis=-1) - points[0]).dot(alpha) + values[0]


def cubic_patch(xi, points, values, grad):
    """Cubic patch interpolation on a simplex

    Interpolation via an D-dimensional cubic polynomial given D+1 points of a
    simplex and their gradients.

    Parameters
    ----------
    xi : tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.
    points : ndarray of floats, shape (D+1, D)
        Data point coordinates.
    values : ndarray of floats, shape (D+1,)
        Data values.
    grad : ndarray of floats, shape (D+1, D)
        Data gradients.

    Returns
    -------
    ndarray
        Array of interpolated values.

    """

    dim = points.shape[1]
    x = [sy.Symbol('x_{}'.format(i)) for i in range(dim)]

    monomials = sy.expand((1 + sum(x))**3).args
    monomials = [m/sy.lambdify(x, m)(*(1 for _ in x)) for m in monomials]

    monomials_f = [sy.lambdify(x, m) for m in monomials]
    monomials_grad = ([sy.lambdify(x, m.diff(x[i])) for m in monomials] for i in range(dim))

    lhs = iter([])
    for fs in chain([monomials_f], monomials_grad):
        lhs = chain(lhs, [np.array([f(*x_val) for f in fs]) for x_val in zip(*np.transpose(points))])
    lhs = np.array(list(lhs))
    rhs = np.ravel([values] + np.transpose(grad).tolist())

    coeffs = np.linalg.pinv(lhs).dot(rhs)

    lbda = sy.lambdify(x, sum((c*m for c, m in zip(coeffs, monomials))), "numpy")
    return lbda(*xi)


def _interp_functor(interpolator, xi, points, *points_props):
    points = np.array(points)
    points_props = tuple(map(np.array, points_props))
    xi = tuple(map(np.array, xi))

    delaunay = (spatial.Delaunay(points) if points.shape[1] >= 2
                else Delaunay1D(points[:, 0]))
    simplex_map = delaunay.find_simplex(np.stack(xi, axis=-1))
    simplex_outer_loc = np.unique(delaunay.convex_hull)

    simplex_outer_base = points[simplex_outer_loc]
    simplex_outer_args = tuple(arg[simplex_outer_loc] for arg in points_props)
    interp = (interpolator(xi, simplex_outer_base, *simplex_outer_args)
              if len(simplex_outer_loc) == points.shape[0] - 1
              else np.full(xi[0].shape, np.nan))

    for i, s in enumerate(delaunay.simplices):
        mask = simplex_map == i
        masked_xi = tuple(dim[mask] for dim in xi)

        simplex_base = points[s]
        simplex_args = tuple(arg[s] for arg in points_props)

        interp[mask] = interpolator(masked_xi, simplex_base, *simplex_args)

    return interp


def interp_levels(points, values, xi):
    """Linear Delaunay interpolation

    Interpolation on D-dimensional scattered points via Delaunay triangulation
    and planar interpolation.

    Parameters
    ----------
    points : ndarray of floats, shape (N, D)
        Data point coordinates.
    values : ndarray of floats, shape (N,)
        Data values.
    xi : tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.

    Returns
    -------
    ndarray
        Array of interpolated values.

    """

    return _interp_functor(plane, xi, points, values)


def interp_1st_order(points, values, grads, xi):
    """1st-order cubic Delaunay interpolation

    Interpolation on D-dimensional scattered points via Delaunay triangulation
    and cubic interpolation with gradient information.

    Parameters
    ----------
    points : ndarray of floats, shape (N, D)
        Data point coordinates.
    values : ndarray of floats, shape (N,)
        Data values.
    grads : ndarray of floats, shape (N, D)
        Data gradients.
    xi : tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.

    Returns
    -------
    ndarray
        Array of interpolated values.

    """

    return _interp_functor(cubic_patch, xi, points, values, grads)
