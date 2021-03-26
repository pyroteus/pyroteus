"""
Utility functions and classes for mesh adaptation.
"""
from __future__ import absolute_import
import firedrake
from firedrake import *
from collections import OrderedDict


def Mesh(arg, **kwargs):
    """
    Overload mesh constructor to endow the output
    mesh with useful quantities.

    The following quantities are computed by default:
        * cell size;
        * facet area.

    Extra quantities can be computed using the flags:
        * `compute_aspect_ratio`;
        * `compute_scaled_jacobian`.

    The argument and keyword arguments are passed to
    the Firedrake `Mesh` constructor, modified so that
    the argument could also be a mesh.
    """
    ar = kwargs.pop('compute_aspect_ratio', False)
    sj = kwargs.pop('compute_scaled_jacobian', False)
    extras = ar
    try:
        mesh = firedrake.Mesh(arg, **kwargs)
    except TypeError:
        mesh = firedrake.Mesh(arg.coordinates, **kwargs)
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.topological_dimension()

    # Cell size
    mesh.delta_x = interpolate(CellSize(mesh), P0)

    # Facet area
    boundary_markers = sorted(mesh.exterior_facets.unique_markers)
    one = Function(P1).assign(1.0)
    if dim == 2:
        mesh.boundary_len = OrderedDict({i: assemble(one*ds(int(i))) for i in boundary_markers})
    else:
        mesh.boundary_area = OrderedDict({i: assemble(one*ds(int(i))) for i in boundary_markers})

    if not extras:
        return mesh
    if dim != 2 or mesh.coordinates.ufl_element().cell() != triangle:
        raise NotImplementedError  # TODO
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = interpolate(Jacobian(mesh), P0_ten)
    edge1 = as_vector([J[0, 0], J[1, 0]])
    edge2 = as_vector([J[0, 1], J[1, 1]])
    edge3 = edge1 - edge2
    a = sqrt(dot(edge1, edge1))
    b = sqrt(dot(edge2, edge2))
    c = sqrt(dot(edge3, edge3))

    # Aspect ratio
    if ar:
        mesh.aspect_ratio = interpolate(a*b*c/((a+b-c)*(b+c-a)*(c+a-b)), P0)

    # Scaled Jacobian
    if sj:
        detJ = JacobianDeterminant(mesh)
        jacobian_sign = sign(detJ)
        max_product = Max(Max(Max(a*b, a*c), Max(b*c, b*a)), Max(c*a, c*b))
        mesh.scaled_jacobian = interpolate(detJ/max_product*jacobian_sign, P0)

    return mesh


def prod(arr):
    """
    Take the product over elements in an array.
    """
    n = len(arr)
    return None if n == 0 else arr[0] if n == 1 else arr[0]*prod(arr[1:])
