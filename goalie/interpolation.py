"""
Driver functions for mesh-to-mesh data transfer.
"""
from .utility import assemble_mass_matrix, cofunction2function, function2cofunction
import firedrake
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py


__all__ = ["project"]


def project(source, target_space, **kwargs):
    r"""
    Overload :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint projection operator
    when applied to :class:`firedrake.cofunction.Cofunction`\s.

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.

    :arg source: the :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` to be projected
    :arg target_space: the :class:`firedrake.functionspaceimpl.FunctionSpace` which we
        seek to project into, or the :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` to use as the target
    """
    if not isinstance(source, (firedrake.Function, firedrake.Cofunction)):
        raise NotImplementedError(
            "Can only currently project Functions and Cofunctions."
        )
    if isinstance(target_space, WithGeometry):
        target = firedrake.Function(target_space)
    elif isinstance(target_space, (firedrake.Cofunction, firedrake.Function)):
        target = target_space
    else:
        raise TypeError(
            "Second argument must be a FunctionSpace, Function, or Cofunction."
        )
    if isinstance(source, firedrake.Cofunction):
        return _project_adjoint(source, target, **kwargs)
    elif source.function_space() == target.function_space():
        return target.assign(source)
    else:
        return _project(source, target, **kwargs)


@PETSc.Log.EventDecorator("goalie.interpolation.project")
def _project(source, target, **kwargs):
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
    Vs = source.function_space()
    Vt = target.function_space()
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                "Source space has multiple components but target space does not."
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                "Inconsistent numbers of components in source and target spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            "Target space has multiple components but source space does not."
        )
    assert isinstance(target, firedrake.Function)
    if hasattr(Vt, "num_sub_spaces"):
        for s, t in zip(source.subfunctions, target.subfunctions):
            t.project(s, **kwargs)
    else:
        target.project(source, **kwargs)
    return target


@PETSc.Log.EventDecorator("goalie.interpolation.project_adjoint")
def _project_adjoint(target_b, source_b, **kwargs):
    """
    Apply the adjoint of a mesh-to-mesh conservative projection to some seed
    :class:`firedrake.cofunction.Cofunction`, mapping to an output
    :class:`firedrake.cofunction.Cofunction`.

    The notation used here is in terms of the adjoint of standard projection.
    However, this function may also be interpreted as a projector in its own right,
    mapping ``target_b`` to ``source_b``.

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.

    :arg target_b: seed :class:`firedrake.cofunction.Cofunction` from the target space
        of the forward projection
    :arg source_b: the :class:`firedrake.cofunction.Cofunction` from the source space
        of the forward projection
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    # Map to Functions to apply the adjoint projection
    if not isinstance(target_b, firedrake.Function):
        target_b = cofunction2function(target_b)
    if not isinstance(source_b, firedrake.Function):
        source_b = cofunction2function(source_b)

    Vt = target_b.function_space()
    Vs = source_b.function_space()
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                "Source space has multiple components but target space does not."
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                "Inconsistent numbers of components in target and source spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            "Target space has multiple components but source space does not."
        )
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    if Vs == Vt:
        source_b.assign(target_b)
    else:
        for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
            ksp = petsc4py.KSP().create()
            ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
            mixed_mass = assemble_mixed_mass_matrix(Vt[i], Vs[i])
            with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
                residual = tb.copy()
                ksp.solveTranspose(tb, residual)
                mixed_mass.mult(residual, sb)  # NOTE: already transposed above

    # Map back to a Cofunction
    return function2cofunction(source_b)
