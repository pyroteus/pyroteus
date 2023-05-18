"""
Driver functions for mesh-to-mesh data transfer.
"""
from .quality import QualityMeasure
from .utility import assemble_mass_matrix, Function, FunctionSpace
import firedrake
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py
from pyop2 import op2
import numpy as np
from typing import Optional
import ufl


__all__ = ["clement_interpolant", "project"]


@PETSc.Log.EventDecorator()
def clement_interpolant(
    source: Function,
    target_space: Optional[FunctionSpace] = None,
    boundary: bool = False,
) -> Function:
    r"""
    Compute the Clement interpolant of a :math:`\mathbb P0` source field, i.e. take the
    volume average over neighbouring cells at each vertex. See :cite:`Cle:75`.

    :arg source: the :math:`\mathbb P0` source field
    :kwarg target_space: the :math:`\mathbb P1` space to interpolate into
    :kwarg boundary: interpolate over boundary facets or cells?
    """

    # Process source space
    Vs = source.function_space()
    Vs_e = Vs.ufl_element()
    if not (Vs_e.family() == "Discontinuous Lagrange" and Vs_e.degree() == 0):
        raise ValueError("Source function provided must be from a P0 space.")
    rank = len(Vs_e.value_shape())
    if rank not in (0, 1, 2):
        raise ValueError(f"Rank-{rank + 1} tensors are not supported.")
    mesh = Vs.mesh()
    dim = mesh.topological_dimension()

    # Process target space
    Vt = target_space
    if Vt is None:
        if rank == 0:
            Vt = FunctionSpace(mesh, "CG", 1)
        elif rank == 1:
            Vt = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        else:
            Vt = firedrake.TensorFunctionSpace(mesh, "CG", 1)
    Vt_e = Vt.ufl_element()
    if not (Vt_e.family() == "Lagrange" and Vt_e.degree() == 1):
        raise ValueError("Target space provided must be P1.")
    target = Function(Vt)

    # Scalar P0 and P1 spaces to hold volumes, etc.
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)

    # Determine target domain
    if rank == 0:
        tdomain = "{[i]: 0 <= i < t.dofs}"
    elif rank == 1:
        tdomain = f"{{[i, j]: 0 <= i < t.dofs and 0 <= j < {dim}}}"
    else:
        tdomain = (
            f"{{[i, j, k]: 0 <= i < t.dofs and 0 <= j < {dim} and 0 <= k < {dim}}}"
        )

    # Compute the patch volume at each vertex
    if not boundary:
        dX = ufl.dx(domain=mesh)
        volume = QualityMeasure(mesh, python=True)("volume")

        # Compute patch volume
        patch_volume = Function(P1)
        domain = "{[i]: 0 <= i < patch.dofs}"
        instructions = "patch[i] = patch[i] + vol[0]"
        keys = {"vol": (volume, op2.READ), "patch": (patch_volume, op2.RW)}
        firedrake.par_loop((domain, instructions), dX, keys, is_loopy_kernel=True)

        # Take weighted average
        if rank == 0:
            instructions = "t[i] = t[i] + v[0] * s[0]"
        elif rank == 1:
            instructions = "t[i, j] = t[i, j] + v[0] * s[0, j]"
        else:
            instructions = f"t[i, {dim} * j + k] = t[i, {dim} * j + k] + v[0] * s[0, {dim} * j + k]"
        keys = {
            "s": (source, op2.READ),
            "v": (volume, op2.READ),
            "t": (target, op2.RW),
        }
        firedrake.par_loop((tdomain, instructions), dX, keys, is_loopy_kernel=True)
    else:
        dX = ufl.ds(domain=mesh)

        # Indicate appropriate boundary
        bnd_indicator = Function(P1)
        firedrake.DirichletBC(P1, 1, "on_boundary").apply(bnd_indicator)

        # Determine facet area for boundary edges
        v = firedrake.TestFunction(P0)
        u = firedrake.TrialFunction(P0)
        bnd_volume = Function(P0)
        mass_term = v * u * dX
        rhs = v * ufl.FacetArea(mesh) * dX
        sp = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "jacobi"}
        firedrake.solve(mass_term == rhs, bnd_volume, solver_parameters=sp)

        # Compute patch volume
        patch_volume = Function(P1)
        domain = "{[i]: 0 <= i < patch.dofs}"
        instructions = "patch[i] = patch[i] + indicator[i] * bnd_vol[0]"
        keys = {
            "bnd_vol": (bnd_volume, op2.READ),
            "indicator": (bnd_indicator, op2.READ),
            "patch": (patch_volume, op2.RW),
        }
        firedrake.par_loop((domain, instructions), dX, keys, is_loopy_kernel=True)

        # Take weighted average
        if rank == 0:
            instructions = "t[i] = t[i] + v[0] * b[i] * s[0]"
        elif rank == 1:
            instructions = "t[i, j] = t[i, j] + v[0] * b[i] * s[0, j]"
        else:
            instructions = f"t[i, {dim} * j + k] = t[i, {dim} * j + k] + v[0] * b[i] * s[0, {dim} * j + k]"
        keys = {
            "s": (source, op2.READ),
            "v": (bnd_volume, op2.READ),
            "b": (bnd_indicator, op2.READ),
            "t": (target, op2.RW),
        }
        firedrake.par_loop((tdomain, instructions), dX, keys, is_loopy_kernel=True)

    # Divide by patch volume and ensure finite
    target.interpolate(target / patch_volume)
    target.dat.data_with_halos[:] = np.nan_to_num(
        target.dat.data_with_halos, posinf=0, neginf=0
    )
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
    if ufl.domain.extract_unique_domain(source) == ufl.domain.extract_unique_domain(
        target
    ):
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
