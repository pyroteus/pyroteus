"""
Driver functions for mesh-to-mesh data transfer.
"""
from .utility import assemble_mass_matrix, cofunction2function, function2cofunction
import firedrake
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py
from typing import Union
import ufl


__all__ = ["project"]


def project(
    source: firedrake.Function,
    target_space: Union[firedrake.FunctionSpace, firedrake.Function],
    adjoint: bool = False,
    **kwargs,
) -> firedrake.Function:
    """
    Overload :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint projection operator.

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.

    :arg source: the :class:`firedrake.function.Function` to be projected
    :arg target_space: the :class:`firedrake.functionspaceimpl.FunctionSpace` which we
        seek to project into
    :kwarg adjoint: apply the transposed projection operator?
    """
    if not isinstance(source, (firedrake.Function, firedrake.Cofunction)):
        raise NotImplementedError(
            "Can only currently project Functions and Cofunctions."
        )  # TODO
    adj_value = isinstance(source, firedrake.Cofunction)
    if adj_value:
        source = cofunction2function(source)
    Vs = source.function_space()
    if isinstance(target_space, firedrake.Function):
        target = target_space
        Vt = target.function_space()
    else:
        Vt = target_space
        target = firedrake.Function(Vt)

    # Account for the case where the meshes match
    source_mesh = ufl.domain.extract_unique_domain(source)
    target_mesh = ufl.domain.extract_unique_domain(target)
    if source_mesh == target_mesh:
        if Vs == Vt:
            return target.assign(source)
        elif not adjoint:
            return target.project(source, **kwargs)

    # Check validity of function spaces
    space1, space2 = ("target", "source") if adjoint else ("source", "target")
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                f"{space1} space has multiple components but {space2} space does not.".capitalize()
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                f"Inconsistent numbers of components in {space1} and {space2} spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            f"{space2} space has multiple components but {space1} space does not.".capitalize()
        )

    # Apply projector
    target = (_project_adjoint if adjoint else _project)(source, target, **kwargs)
    if adj_value:
        target = function2cofunction(target)
    return target


@PETSc.Log.EventDecorator("goalie.interpolation.project")
def _project(
    source: firedrake.Function, target: firedrake.Function, **kwargs
) -> firedrake.Function:
    """
    Apply a mesh-to-mesh conservative projection to some source
    :class:`firedrake.function.Function`, mapping to a target
    :class:`firedrake.function.Function`.

    This function extends to the case of mixed spaces.

    Extra keyword arguments are passed to Firedrake's
    :func:`firedrake.projection.project`` function.

    :arg source: the `Function` to be projected
    :arg target: the `Function` which we seek to project onto
    """
    assert isinstance(target, firedrake.Function)
    if hasattr(target.function_space(), "num_sub_spaces"):
        assert hasattr(source.function_space(), "num_sub_spaces")
        for s, t in zip(source.subfunctions, target.subfunctions):
            t.project(s, **kwargs)
    else:
        target.project(source, **kwargs)
    return target


@PETSc.Log.EventDecorator("goalie.interpolation.project_adjoint")
def _project_adjoint(
    target_b: firedrake.Function, source_b: firedrake.Function, **kwargs
) -> firedrake.Function:
    """
    Apply the adjoint of a mesh-to-mesh conservative projection to some seed
    :class:`firedrake.function.Function`, mapping to an output
    :class:`firedrake.function.Function`.

    The notation used here is in terms of the adjoint of standard projection.
    However, this function may also be interpreted as a projector in its own right,
    mapping ``target_b`` to ``source_b``.

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.

    :arg target_b: seed :class:`firedrake.function.Function` from the target space of
        the forward projection
    :arg source_b: the :class:`firedrake.function.Function` from the source space of
        the forward projection
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    Vt = target_b.function_space()
    assert isinstance(source_b, firedrake.Function)
    Vs = source_b.function_space()

    # Get subspaces
    if hasattr(Vs, "num_sub_spaces"):
        assert hasattr(Vt, "num_sub_spaces")
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
        ksp = petsc4py.KSP().create()
        ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
        mixed_mass = assemble_mixed_mass_matrix(Vt[i], Vs[i])
        with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
            residual = tb.copy()
            ksp.solveTranspose(tb, residual)
            mixed_mass.mult(residual, sb)  # NOTE: mixed mass already transposed

    return source_b
