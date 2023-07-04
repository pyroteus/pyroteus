"""
Functions used frequently for testing.
"""
import firedrake
from pyroteus.metric import RiemannianMetric
import ufl


def uniform_mesh(dim, n, l=1, **kwargs):
    args = [n] * dim + [l]
    return (firedrake.SquareMesh if dim == 2 else firedrake.CubeMesh)(*args, **kwargs)


def uniform_metric(function_space, scaling):
    dim = function_space.mesh().topological_dimension()
    metric = RiemannianMetric(function_space)
    metric.interpolate(scaling * ufl.Identity(dim))
    return metric
