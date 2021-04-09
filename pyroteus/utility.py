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


def norm(v, norm_type='L2', mesh=None):
    r"""
    Overload Firedrake's `norm` function to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive, i.e. l2 and L2 will give different
    results in general.
    """
    norm_codes = {'l1': 0, 'l2': 2, 'linf': 3}
    if norm_type in norm_codes:
        with v.dat.vec_ro as vv:
            return vv.norm(norm_codes[norm_type])
    elif norm_type[0] == 'l':
        raise NotImplementedError("lp norm of order {:s} not supported.".format(norm_type[1:]))
    else:
        return firedrake.norm(v, norm_type=norm_type, mesh=mesh)


def errornorm(u, uh, norm_type='L2', **kwargs):
    r"""
    Overload Firedrake's `errornorm` function to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive, i.e. l2 and L2 will give different
    results in general.
    """
    if len(u.ufl_shape) != len(uh.ufl_shape):
        raise RuntimeError("Mismatching rank between u and uh")

    if not isinstance(uh, function.Function):
        raise ValueError("uh should be a Function, is a %r", type(uh))
    if norm_type[0] == 'l':
        if not isinstance(u, function.Function):
            raise ValueError("u should be a Function, is a %r", type(uh))

    if isinstance(u, firedrake.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            firedrake.logging.warning("Degree of exact solution less than approximation degree")

    if norm_type[0] == 'l':  # Point-wise norms
        v = u
        v -= uh
    else:  # Norms in UFL
        v = u - uh
    return norm(v, norm_type=norm_type, **kwargs)


def rotation_matrix_2d(angle):
    """
    Rotation matrix associated with some
    `angle`, as a UFL matrix.
    """
    return as_matrix([[cos(angle), -sin(angle)],
                     [sin(angle), cos(angle)]])


def rotate(v, angle, origin=None):
    """
    Rotate a UFL :class:`as_vector` `v`
    by `angle` about an `origin`.
    """
    dim = len(v)
    origin = origin or as_vector(np.zeros(dim))
    assert len(origin) == dim, "Origin does not match dimension"
    if dim == 2:
        R = rotation_matrix_2d(angle)
    else:
        raise NotImplementedError
    return dot(R, v - origin) + origin
