"""
Driver functions for mesh-to-mesh data transfer.
"""
from .utility import assemble_mass_matrix, QualityMeasure, Function, FunctionSpace
import firedrake
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py
from pyop2 import op2
import ufl
import numpy as np


__all__ = ["clement_interpolant", "project"]


@PETSc.Log.EventDecorator("pyroteus.clement_interpolant")
def clement_interpolant(source: Function, **kwargs) -> Function:
    r"""
    Compute the Clement interpolant of a :math:`\mathbb P0`
    source field, i.e. take the volume average over
    neighbouring cells at each vertex. See :cite:`Cle:75`.

    :arg source: the :math:`\mathbb P0` source field
    :kwarg target_space: the :math:`\mathbb P1` space to
        interpolate into
    :boundary_tag: optional boundary tag to compute the
        Clement interpolant over.
    """
    target_space = kwargs.get("target_space")
    boundary_tag = kwargs.get("boundary_tag")
    V = source.function_space()
    assert V.ufl_element().family() == "Discontinuous Lagrange"
    assert V.ufl_element().degree() == 0
    rank = len(V.ufl_element().value_shape())
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    P1 = FunctionSpace(mesh, "CG", 1)
    dX = ufl.dx if boundary_tag is None else ufl.ds(boundary_tag)
    if target_space is None:
        if rank == 0:
            target_space = P1
        elif rank == 1:
            target_space = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        elif rank == 2:
            target_space = firedrake.TensorFunctionSpace(mesh, "CG", 1)
        else:
            raise ValueError(f"Rank-{rank} tensors are not supported.")
    else:
        assert target_space.ufl_element().family() == "Lagrange"
        assert target_space.ufl_element().degree() == 1
    target = Function(target_space)

    # Compute the patch volume at each vertex
    if boundary_tag is None:
        P0 = FunctionSpace(mesh, "DG", 0)
        dx = ufl.dx(domain=mesh)
        volume = firedrake.assemble(firedrake.TestFunction(P0) * dx)
    else:
        volume = QualityMeasure(mesh, python=True)("facet_area")
    patch_volume = Function(P1)
    kernel = "for (int i=0; i < p.dofs; i++) p[i] += v[0];"
    keys = {
        "v": (volume, op2.READ),
        "p": (patch_volume, op2.INC),
    }
    firedrake.par_loop(kernel, dX, keys)

    # Volume average
    keys = {
        "s": (source, op2.READ),
        "v": (volume, op2.READ),
        "t": (target, op2.INC),
    }
    if rank == 0:
        firedrake.par_loop(
            """
            for (int i=0; i < t.dofs; i++) {
              t[i] += s[0]*v[0];
            }
            """,
            dX,
            keys,
        )
    elif rank == 1:
        firedrake.par_loop(
            """
            int d = %d;
            for (int i=0; i < t.dofs; i++) {
              for (int j=0; j < d; j++) {
                t[i*d + j] += s[j]*v[0];
              }
            }
            """
            % dim,
            dX,
            keys,
        )
    elif rank == 2:
        firedrake.par_loop(
            """
            int d = %d;
            int Nd = d*d;
            for (int i=0; i < t.dofs; i++) {
              for (int j=0; j < d; j++) {
                for (int k=0; k < d; k++) {
                  t[i*Nd + j*d + k] += s[j*d + k]*v[0];
                }
              }
            }
            """
            % dim,
            dX,
            keys,
        )
    else:
        raise ValueError(f"Rank-{rank} tensors are not supported.")
    target.interpolate(target / patch_volume)
    if boundary_tag is not None:
        target.dat.data_with_halos[:] = np.nan_to_num(target.dat.data_with_halos)
    return target


# --- Linear interpolation

# TODO

# --- Conservative interpolation by supermesh projection


def project(
    source: Function,
    target_space: FunctionSpace,
    adjoint: bool = False,
    **kwargs,
) -> Function:
    """
    Overload :func:`firedrake.projection.project` to account
    for the case of two mixed function spaces defined on
    different meshes and for the adjoint projection
    operator.

    Extra keyword arguments are passed to
    :func:`firedrake.projection.project`.

    :arg source: the :class:`firedrake.function.Function`
        to be projected
    :arg target_space: the
        :class:`firedrake.functionspaceimpl.FunctionSpace`
        which we seek to project into
    :kwarg adjoint: apply the transposed projection
        operator?
    """
    if not isinstance(source, Function):
        raise NotImplementedError("Can only currently project Functions")  # TODO
    source_space = source.function_space()
    if isinstance(target_space, Function):
        target = target_space
        target_space = target.function_space()
    else:
        target = Function(target_space)
    if source_space.ufl_domain() == target_space.ufl_domain():
        if source_space == target_space:
            target.assign(source)
        else:
            target.project(source, **kwargs)
        return target
    else:
        return mesh2mesh_project(source, target, adjoint=adjoint, **kwargs)


@PETSc.Log.EventDecorator("pyroteus.mesh2mesh_project")
def mesh2mesh_project(
    source: Function,
    target: Function,
    adjoint: bool = False,
    **kwargs,
) -> Function:
    """
    Apply a mesh-to-mesh conservative projection to some
    ``source``, mapping to a ``target``.

    This function extends to the case of mixed spaces.

    Extra keyword arguments are passed to Firedrake's
    ``project`` function.

    :arg source: the :class:`firedrake.function.Function`
        to be projected
    :arg target: the :class:`firedrake.function.Function`
        which we seek to project onto
    :kwarg adjoint: apply the transposed projection
        operator?
    """
    if adjoint:
        return mesh2mesh_project_adjoint(source, target)
    source_space = source.function_space()
    assert isinstance(target, Function)
    target_space = target.function_space()
    if source_space == target_space:
        target.assign(source)
    elif hasattr(target_space, "num_sub_spaces"):
        assert hasattr(source_space, "num_sub_spaces")
        assert target_space.num_sub_spaces() == source_space.num_sub_spaces()
        for s, t in zip(source.subfunctions, target.subfunctions):
            t.project(s, **kwargs)
    else:
        target.project(source, **kwargs)
    return target


@PETSc.Log.EventDecorator("pyroteus.mesh2mesh_project_adjoint")
def mesh2mesh_project_adjoint(
    target_b: Function, source_b: Function, **kwargs
) -> Function:
    """
    Apply the adjoint of a mesh-to-mesh conservative
    projection to some seed ``target_b``, mapping to
    ``source_b``.

    The notation used here is in terms of the adjoint of
    ``mesh2mesh_project``. However, this function may also
    be interpreted as a projector in its own right,
    mapping ``target_b`` to ``source_b``.

    Extra keyword arguments are passed to
    :func:`firedrake.projection.project`.

    :arg target_b: seed :class:`firedrake.function.Function`
        from the target space of the forward projection
    :arg source_b: the :class:`firedrake.function.Function`
        from the source space of the forward projection
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    target_space = target_b.function_space()
    assert isinstance(source_b, Function)
    source_space = source_b.function_space()

    # Get subspaces
    if source_space == target_space:
        source_b.assign(target_b)
        return source_b
    elif hasattr(source_space, "num_sub_spaces"):
        if not hasattr(target_space, "num_sub_spaces"):
            raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
        if not target_space.num_sub_spaces() == source_space.num_sub_spaces():
            raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    elif hasattr(target_space, "num_sub_spaces"):
        raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
        ksp = petsc4py.KSP().create()
        ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
        mixed_mass = assemble_mixed_mass_matrix(
            t_b.function_space(), s_b.function_space()
        )
        with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
            residual = tb.copy()
            ksp.solveTranspose(tb, residual)
            mixed_mass.mult(residual, sb)  # NOTE: mixed mass already transposed

    return source_b
