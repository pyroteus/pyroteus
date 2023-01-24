"""
Utility functions and classes for mesh adaptation.
"""
from .quality import *
import mpi4py
import petsc4py
from .log import *
from collections import OrderedDict
import numpy as np


@PETSc.Log.EventDecorator("pyroteus.Mesh")
def Mesh(arg, **kwargs) -> MeshGeometry:
    """
    Overload :func:`firedrake.mesh.Mesh` to
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
    bnd_len = OrderedDict(
        {i: firedrake.assemble(one * ufl.ds(int(i))) for i in boundary_markers}
    )
    if dim == 2:
        mesh.boundary_len = bnd_len
    else:
        mesh.boundary_area = bnd_len

    # Cell size
    if dim == 2 and mesh.coordinates.ufl_element().cell() == ufl.triangle:
        mesh.delta_x = firedrake.interpolate(ufl.CellDiameter(mesh), P0)

    return mesh


class File(firedrake.output.File):
    """
    Overload :class:`firedrake.output.File` so that
    it uses ``adaptive`` mode by default. Whilst
    this means that the mesh topology is
    recomputed at every export, it removes any
    need for the user to reset it manually.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("adaptive", True)
        super().__init__(*args, **kwargs)

    def _write_vtu(self, *functions):
        """
        Overload the Firedrake functionality
        under the blind assumption that the
        same list of functions are outputted
        each time (albeit on different meshes).
        """
        if self._fnames is not None:
            if len(self._fnames) != len(functions):
                raise ValueError("Writing different set of functions")
            for name, f in zip(self._fnames, functions):
                if f.name() != name:
                    f.rename(name)
        return super()._write_vtu(*functions)


@PETSc.Log.EventDecorator("pyroteus.assemble_mass_matrix")
def assemble_mass_matrix(
    space: FunctionSpace, norm_type: str = "L2"
) -> petsc4py.PETSc.Mat:
    """
    Assemble the ``norm_type`` mass matrix
    associated with some finite element ``space``.
    """
    trial = firedrake.TrialFunction(space)
    test = firedrake.TestFunction(space)
    if norm_type == "L2":
        lhs = ufl.inner(trial, test) * ufl.dx
    elif norm_type == "H1":
        lhs = (
            ufl.inner(trial, test) * ufl.dx
            + ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        )
    else:
        raise ValueError(f"Norm type '{norm_type}' not recognised.")
    return firedrake.assemble(lhs).petscmat


@PETSc.Log.EventDecorator("pyroteus.norm")
def norm(v: Function, norm_type: str = "L2", **kwargs) -> float:
    r"""
    Overload :func:`firedrake.norms.norm` to
    allow for :math:`\ell^p` norms.

    Note that this version is case sensitive,
    i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg v: the :class:`firedrake.function.Function`
        to take the norm of
    :kwarg norm_type: choose from ``'l1'``, ``'l2'``,
        ``'linf'``, ``'L2'``, ``'Linf'``, ``'H1'``,
        ``'Hdiv'``, ``'Hcurl'``, or any ``'Lp'`` with
        :math:`p >= 1`.
    :kwarg condition: a UFL condition for specifying
        a subdomain to compute the norm over
    :kwarg boundary: should the norm be computed over
        the domain boundary?
    """
    boundary = kwargs.get("boundary", False)
    condition = kwargs.get("condition", firedrake.Constant(1.0))
    norm_codes = {"l1": 0, "l2": 2, "linf": 3}
    p = 2
    if norm_type in norm_codes or norm_type == "Linf":
        if boundary:
            raise NotImplementedError("lp errors on the boundary not yet implemented.")
        v.interpolate(condition * v)
        if norm_type == "Linf":
            with v.dat.vec_ro as vv:
                return vv.max()[1]
        else:
            with v.dat.vec_ro as vv:
                return vv.norm(norm_codes[norm_type])
    elif norm_type[0] == "l":
        raise NotImplementedError(
            "lp norm of order {:s} not supported.".format(norm_type[1:])
        )
    else:
        dX = ufl.ds if boundary else ufl.dx
        if norm_type.startswith("L"):
            try:
                p = int(norm_type[1:])
            except Exception:
                raise ValueError(f"Don't know how to interpret '{norm_type}' norm.")
            if p < 1:
                raise ValueError(f"'{norm_type}' norm does not make sense.")
            integrand = ufl.inner(v, v)
        elif norm_type.lower() == "h1":
            integrand = ufl.inner(v, v) + ufl.inner(ufl.grad(v), ufl.grad(v))
        elif norm_type.lower() == "hdiv":
            integrand = ufl.inner(v, v) + ufl.div(v) * ufl.div(v)
        elif norm_type.lower() == "hcurl":
            integrand = ufl.inner(v, v) + ufl.inner(ufl.curl(v), ufl.curl(v))
        else:
            raise ValueError(f"Unknown norm type '{norm_type}'.")
        return firedrake.assemble(condition * integrand ** (p / 2) * dX) ** (1 / p)


@PETSc.Log.EventDecorator("pyroteus.errornorm")
def errornorm(u, uh: Function, norm_type: str = "L2", **kwargs) -> float:
    r"""
    Overload :func:`firedrake.norms.errornorm`
    to allow for :math:`\ell^p` norms.

    Note that this version is case sensitive,
    i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg u: the 'true' value
    :arg uh: the approximation of the 'truth'
    :kwarg norm_type: choose from ``'l1'``, ``'l2'``,
        ``'linf'``, ``'L2'``, ``'Linf'``, ``'H1'``,
        ``'Hdiv'``, ``'Hcurl'``, or any ``'Lp'`` with
        :math:`p >= 1`.
    :kwarg boundary: should the norm be computed over
        the domain boundary?
    """
    if len(u.ufl_shape) != len(uh.ufl_shape):
        raise RuntimeError("Mismatching rank between u and uh.")

    if not isinstance(uh, Function):
        raise TypeError(f"uh should be a Function, is a {type(uh).__name__}.")
    if norm_type[0] == "l":
        if not isinstance(u, Function):
            raise TypeError(f"u should be a Function, is a {type(u).__name__}.")

    if isinstance(u, Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            firedrake.logging.warning(
                "Degree of exact solution less than approximation degree"
            )

    # Case 1: point-wise norms
    if norm_type[0] == "l":
        v = u
        v -= uh

    # Case 2: UFL norms for mixed function spaces
    elif hasattr(uh.function_space(), "num_sub_spaces"):
        if norm_type == "L2":
            vv = [uu - uuh for uu, uuh in zip(u.split(), uh.split())]
            dX = ufl.ds if kwargs.get("boundary", False) else ufl.dx
            return ufl.sqrt(firedrake.assemble(sum([ufl.inner(v, v) for v in vv]) * dX))
        else:
            raise NotImplementedError(
                f"Norm type '{norm_type}' not supported for mixed spaces."
            )

    # Case 3: UFL norms for non-mixed spaces
    else:
        v = u - uh

    return norm(v, norm_type=norm_type, **kwargs)


class AttrDict(dict):
    """
    Dictionary that provides both ``self[key]``
    and ``self.key`` access to members.

    **Disclaimer**: Copied from `stackoverflow
    <http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python>`__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def effectivity_index(error_indicator: Function, Je: float) -> float:
    r"""
    Overestimation factor of some error estimator
    for the QoI error.

    Note that this is only typically used for simple
    steady-state problems with analytical solutions.

    :arg error_indicator: a :math:`\mathbb P0`
        :class:`firedrake.function.Function` which
        localises contributions to an error estimator
        to individual elements
    :arg Je: error in quantity of interest
    """
    if not isinstance(error_indicator, Function):
        raise ValueError("Error indicator must return a Function.")
    el = error_indicator.ufl_element()
    if not (el.family() == "Discontinuous Lagrange" and el.degree() == 0):
        raise ValueError("Error indicator must be P0.")
    eta = error_indicator.vector().gather().sum()
    return np.abs(eta / Je)


def create_directory(
    path: str, comm: mpi4py.MPI.Intracomm = firedrake.COMM_WORLD
) -> str:
    """
    Create a directory on disk.

    Code copied from `Thetis
    <https://thetisproject.org>`__.

    :arg path: path to the directory
    :kwarg comm: MPI communicator
    """
    if comm.rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    comm.barrier()
    return path
