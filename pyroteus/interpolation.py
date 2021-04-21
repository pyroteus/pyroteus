from firedrake import *


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
        return target_b
    elif hasattr(source_space, 'num_sub_spaces'):
        assert hasattr(target_space, 'num_sub_spaces')
        assert target_space.num_sub_spaces() == source_space.num_sub_spaces()
        target_b_split = target_b.split()
        source_b_split = source_b.split()
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    for t_b, s_b in zip(target_b_split, source_b_split):

        # Assemble mass matrix for target space and apply its inverse
        M_t = assemble(inner(TrialFunction(target_space), TestFunction(target_space))*dx).M.handle
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_t.transpose())
        ksp.setFromOptions()
        with t_b.dat.vec_ro as tb:
            residual = tb.copy()
            ksp.solve(tb, residual)

        # Assemble mixed mass matrix and multiply by its transpose
        M_st = assemble_mixed_mass_matrix(source_space, target_space)
        with s_b.dat.vec_ro as sb:
            M_st.multTranspose(residual, sb)

    return source_b
