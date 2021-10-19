"""
Utility functions and classes for mesh adaptation.
"""
from __future__ import absolute_import
from pyroteus.mesh_quality import *
import firedrake
from .log import *
from collections import OrderedDict
from collections.abc import Iterable
import firedrake.cython.dmcommon as dmcommon


def Mesh(arg, **kwargs):
    """
    Overload Firedrake's ``Mesh`` constructor to
    endow the output mesh with useful quantities.

    The following quantities are computed by default:
        * cell size;
        * facet area.

    The argument and keyword arguments are passed to
    Firedrake's ``Mesh`` constructor, modified so
    that the argument could also be a mesh.
    """
    try:
        mesh = firedrake.Mesh(arg, **kwargs)
    except TypeError:
        mesh = firedrake.Mesh(arg.coordinates, **kwargs)
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.topological_dimension()

    # Facet area
    boundary_markers = sorted(mesh.exterior_facets.unique_markers)
    one = Function(P1).assign(1.0)
    if dim == 2:
        mesh.boundary_len = OrderedDict({i: assemble(one*ds(int(i))) for i in boundary_markers})
    else:
        mesh.boundary_area = OrderedDict({i: assemble(one*ds(int(i))) for i in boundary_markers})

    # Cell size
    if dim == 2 and mesh.coordinates.ufl_element().cell() == triangle:
        mesh.delta_x = interpolate(CellSize(mesh), P0)

    return mesh


class File(firedrake.output.File):
    """
    Overload Firedrake's ``File`` class so that
    it uses ``adaptive`` mode by default. Whilst
    this means that the mesh topology is
    recomputed at every export, it removes any
    need for the user to reset it manually.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('adaptive', True)
        super(File, self).__init__(*args, **kwargs)

    def _write_vtu(self, *functions):
        """
        Overload the Firedrake functionality
        under the blind assumption that the
        same list of functions are outputted
        each time (albeit on different meshes).
        """
        if self._fnames is not None:
            assert len(self._fnames) == len(functions), "Writing different set of functions"
            for name, f in zip(self._fnames, functions):
                if f.name() != name:
                    f.rename(name)
        return super(File, self)._write_vtu(*functions)


def prod(arr):
    """
    Take the product over elements in an array.
    """
    n = len(arr)
    return None if n == 0 else arr[0] if n == 1 else arr[0]*prod(arr[1:])


def assemble_mass_matrix(space, norm_type='L2'):
    """
    Assemble the ``norm_type`` mass matrix
    associated with some finite element ``space``.
    """
    trial = TrialFunction(space)
    test = TestFunction(space)
    if norm_type == 'L2':
        lhs = inner(trial, test)*dx
    elif norm_type == 'H1':
        lhs = inner(trial, test)*dx + inner(grad(trial), grad(test))*dx
    else:
        raise ValueError(f"Norm type {norm_type} not recognised.")
    return assemble(lhs).petscmat


def norm(v, norm_type='L2', mesh=None, condition=None, boundary=False):
    r"""
    Overload Firedrake's ``norm`` function to
    allow for :math:`\ell^p` norms.

    Note that this version is case sensitive,
    i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg v: the :class:`Function` to take the norm of
    :kwarg norm_type: choose from 'l1', 'l2', 'linf',
        'L2', 'Linf', 'H1', 'Hdiv', 'Hcurl', or any
        'Lp' with :math:`p >= 1`.
    :kwarg mesh: the mesh that `v` is defined upon
    :kwarg condition: a UFL condition for specifying
        a subdomain to compute the norm over
    :kwarg boundary: should the norm be computed over
        the domain boundary?
    """
    norm_codes = {'l1': 0, 'l2': 2, 'linf': 3}
    if norm_type in norm_codes or norm_type == 'Linf':
        if boundary:
            raise NotImplementedError("lp errors on the boundary not yet implemented.")
        if condition is not None:
            v.interpolate(condition*v)
        if norm_type == 'Linf':
            with v.dat.vec_ro as vv:
                return vv.max()[1]
        else:
            with v.dat.vec_ro as vv:
                return vv.norm(norm_codes[norm_type])
    elif norm_type[0] == 'l':
        raise NotImplementedError("lp norm of order {:s} not supported.".format(norm_type[1:]))
    else:
        condition = condition or Constant(1.0)
        dX = ds if boundary else dx
        if norm_type.startswith('L'):
            try:
                p = int(norm_type[1:])
                if p < 1:
                    raise ValueError(f"{norm_type} norm does not make sense.")
            except ValueError:
                raise ValueError(f"Don't know how to interpret {norm_type} norm.")
            return assemble(condition*inner(v, v)**(p/2)*dX)**(1/p)
        elif norm_type.lower() == 'h1':
            return sqrt(assemble(condition*(inner(v, v) + inner(grad(v), grad(v)))*dX))
        elif norm_type.lower() == 'hdiv':
            return sqrt(assemble(condition*(inner(v, v) + div(v)*div(v))*dX))
        elif norm_type.lower() == 'hcurl':
            return sqrt(assemble(condition*(inner(v, v) + inner(curl(v), curl(v)))*dX))
        else:
            raise ValueError(f"Unknown norm type {norm_type}")


def errornorm(u, uh, norm_type='L2', **kwargs):
    r"""
    Overload Firedrake's ``errornorm`` function
    to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive,
    i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.
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

    # Case 1: point-wise norms
    if norm_type[0] == 'l':
        v = u
        v -= uh

    # Case 2: UFL norms for mixed function spaces
    elif hasattr(uh.function_space(), 'num_sub_spaces'):
        if norm_type[1:] == '2':
            vv = [uu - uuh for uu, uuh in zip(u.split(), uh.split())]
            return sqrt(assemble(sum([inner(v, v) for v in vv])*dx))
        else:
            raise NotImplementedError

    # Case 3: UFL norms for non-mixed spaces
    else:
        v = u - uh

    return norm(v, norm_type=norm_type, **kwargs)


def rotation_matrix_2d(angle):
    """
    Rotation matrix associated with some
    ``angle``, as a UFL matrix.
    """
    return as_matrix([[cos(angle), -sin(angle)],
                     [sin(angle), cos(angle)]])


def rotate(v, angle, origin=None):
    """
    Rotate a UFL :class:`as_vector` ``v``
    by ``angle`` about an ``origin``.
    """
    dim = len(v)
    origin = origin or as_vector(np.zeros(dim))
    assert len(origin) == dim, "Origin does not match dimension"
    if dim == 2:
        R = rotation_matrix_2d(angle)
    else:
        raise NotImplementedError
    return dot(R, v - origin) + origin


class AttrDict(dict):
    """
    Dictionary that provides both ``self['key']``
    and ``self.key`` access to members.

    **Disclaimer**: Copied from `stackoverflow <http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python>`__.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def effectivity_index(error_indicator, Je):
    r"""
    Overestimation factor of some error estimator
    for the QoI error.

    Note that this is only typically used for simple
    steady-state problems with analytical solutions.

    :arg error_indicator: a :math:`\mathbb P0`
        :class:`Function` which localises
        contributions to an error estimator to
        individual elements
    :arg Je: error in quantity of interest
    """
    assert isinstance(error_indicator, Function), "Error indicator must return a Function"
    el = error_indicator.ufl_element()
    assert (el.family(), el.degree()) == ('Discontinuous Lagrange', 0), "Error indicator must be P0"
    eta = error_indicator.vector().gather().sum()
    return np.abs(eta/Je)


def classify_element(element, dim):
    """
    Classify a :class:`FiniteElement` in terms
    of a label and a list of entity DOFs.

    :arg element: the :class:`FiniteElement`
    :arg dim: the topological dimension
    """
    p = element.degree()
    family = element.family()
    n = len(element.sub_elements()) or 1
    label = {1: '', dim: 'Vector ', dim**2: 'Tensor '}[n]
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    if family == 'Discontinuous Lagrange' and p == 0:
        entity_dofs[-1] = n
        label += f"P{p}DG"
    elif family == 'Lagrange' and p == 1:
        entity_dofs[0] = n
        label += f"P{p}"
    else:
        raise NotImplementedError
    return label, entity_dofs


def create_section(mesh, element):
    """
    Create a PETSc section associated with
    a mesh and some :class:`FiniteElement`.

    :arg mesh: the mesh
    :arg element: the :class:`FiniteElement`
        for which a section is sought, or a
        list of entity DOFs to be passed
        straight through
    """
    if isinstance(element, Iterable):
        return dmcommon.create_section(mesh, element)
    else:
        dim = mesh.topological_dimension()
        label, entity_dofs = classify_element(element, dim)
        return dmcommon.create_section(mesh, entity_dofs)


def create_directory(path, comm=firedrake.COMM_WORLD):
    """
    Create a directory on disk.

    :arg path: path to the directory
    :kwarg comm: MPI communicator
    """
    if comm.rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    comm.barrier()
    return path
