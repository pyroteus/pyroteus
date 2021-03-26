"""
Utility functions and classes for mesh adaptation.
"""
from __future__ import absolute_import
import firedrake
from firedrake import *


def Mesh(*args, **kwargs):
    """
    Overload mesh constructor to endow the output
    mesh with useful statistics:
        * :attr:`delta_x` cell size.
    """
    mesh = firedrake.Mesh(*args, **kwargs)
    P0 = FunctionSpace(mesh, "DG", 0)
    mesh.delta_x = interpolate(CellSize(mesh), P0)
    return mesh
