"""
Functions used frequently for testing.
"""
import firedrake


def uniform_mesh(dim, n, l=1):
    return firedrake.SquareMesh(n, n, l) if dim == 2 else firedrake.CubeMesh(n, n, n, l)
