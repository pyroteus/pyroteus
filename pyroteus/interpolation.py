from __future__ import absolute_import
from .utility import *


__all__ = ["mesh2mesh_project", "mesh2mesh_project_adjoint"]


def mesh2mesh_project(source, target_space, adjoint=False, **kwargs):
    """
    Apply a mesh-to-mesh conservative projection to some
    `source`, mapping into a `target_space`.

    This function extends to the case of mixed spaces.

    :arg source: the :class:`Function` to be projected
    :arg target_space: the :class:`FunctionSpace` which we
        seek to project into
    """
    if adjoint:
        return mesh2mesh_project_adjoint(source, target_space)
    source_space = source.function_space()
    if source_space == target_space:
        return source
    elif hasattr(target_space, 'num_sub_spaces'):
        assert hasattr(source_space, 'num_sub_spaces')
        assert target_space.num_sub_spaces() == source_space.num_sub_spaces()
        target = Function(target_space)
        for s, t in zip(source.split(), target.split()):
            t.project(s, **kwargs)
        return target
    else:
        return project(source, target_space, **kwargs)


def mesh2mesh_project_adjoint(target_b, source_space, **kwargs):
    """
    Apply the adjoint of a mesh-to-mesh conservative
    projection to some seed `target_b`, mapping into a
    `source_space`.

    The notation used here is in terms of the adjoint of
    `mesh2mesh_project`. However, this function may also
    be interpreted as a projector in its own right,
    mapping `target_b` into `source_space`.

    :arg target_b: seed :class:`Function` from the target
        space
    :arg source_space: the :class:`FunctionSpace` which
        the forward projection maps from
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    target_space = target_b.function_space()
    source_b = Function(source_space)

    # Get subspaces
    if source_space == target_space:
        source_b.assign(target_b)
        return source_b
    elif hasattr(source_space, 'num_sub_spaces'):
        if not hasattr(target_space, 'num_sub_spaces'):
            raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
        if not target_space.num_sub_spaces() == source_space.num_sub_spaces():
            raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
        target_b_split = target_b.split()
        source_b_split = source_b.split()
    elif hasattr(target_space, 'num_sub_spaces'):
        raise ValueError(f"Incompatible spaces {source_space} and {target_space}")
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
        ksp = PETSc.KSP().create()
        ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
        mixed_mass = assemble_mixed_mass_matrix(t_b.function_space(), s_b.function_space())
        with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
            residual = tb.copy()
            ksp.solveTranspose(tb, residual)
            mixed_mass.mult(residual, sb)  # NOTE: mixed mass already transposed

    return source_b
