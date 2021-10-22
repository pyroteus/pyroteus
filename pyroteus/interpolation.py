"""
Driver functions for mesh-to-mesh data transfer.
"""
from __future__ import absolute_import
from .utility import *


__all__ = ["clement_interpolant", "project"]


def clement_interpolant(source, boundary_tag=None):
    r"""
    Compute the Clement interpolant of a :math:`\mathbb P0`
    source field, i.e. take the volume average over
    neighbouring cells at each vertex.

    :arg source: the :math:`\mathbb P0` source field
    """
    V = source.function_space()
    assert V.ufl_element().family() == 'Discontinuous Lagrange'
    assert V.ufl_element().degree() == 0
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    P1 = FunctionSpace(mesh, "CG", 1)
    dX = dx if boundary_tag is None else ds(boundary_tag)

    # Compute the patch volume at each vertex
    if boundary_tag is None:
        P0 = FunctionSpace(mesh, "DG", 0)
        volume = assemble(TestFunction(P0)*dx(domain=mesh))
    else:
        volume = get_facet_areas(mesh)
    patch_volume = Function(P1)
    kernel = "for (int i=0; i < p.dofs; i++) p[i] += v[0];"
    par_loop(kernel, dX, {'v': (volume, READ), 'p': (patch_volume, INC)})

    # Volume average
    rank = len(V.ufl_element().value_shape())
    if rank == 0:
        target = Function(P1)
        par_loop("for (int i=0; i < t.dofs; i++) t[i] += s[0]*v[0];", dX,
                 {'s': (source, READ), 'v': (volume, READ), 't': (target, INC)})
        target.interpolate(target/patch_volume)
    elif rank == 1:
        target = Function(VectorFunctionSpace(mesh, "CG", 1))
        par_loop("""
            int d = %d;
            for (int i=0; i < t.dofs; i++) {
              for (int j=0; j < d; j++) {
                t[i*d + j] += s[j]*v[0];
              }
            }
            """ % dim, dX, {'s': (source, READ), 'v': (volume, READ), 't': (target, INC)})
        target.interpolate(target/patch_volume)
    elif rank == 2:
        target = Function(TensorFunctionSpace(mesh, "CG", 1))
        par_loop("""
            int d = %d;
            int Nd = d*d;
            for (int i=0; i < t.dofs; i++) {
              for (int j=0; j < d; j++) {
                for (int k=0; k < d; k++) {
                  t[i*Nd + j*d + k] += s[j*d + k]*v[0];
                }
              }
            }
            """ % dim, dX, {'s': (source, READ), 'v': (volume, READ), 't': (target, INC)})
        target.interpolate(target/patch_volume)
    else:
        raise ValueError(f"Rank-{rank} tensors are not supported.")
    return target


# --- Linear interpolation

# TODO

# --- Conservative interpolation by supermesh projection

def project(source, target_space, adjoint=False, **kwargs):
    """
    Overload Firedrake's ``project`` function to account
    for the case of two mixed function spaces defined on
    different meshes and for the adjoint projection
    operator.

    Extra keyword arguments are passed to Firedrake's
    ``project`` function.

    :arg source: the :class:`Function` to be projected
    :arg target_space: the :class:`FunctionSpace` which
        we seek to project into
    :kwarg adjoint: apply the transposed projection
        operator?
    """
    if not isinstance(source, firedrake.Function):
        raise NotImplementedError("Can only currently project Functions")  # TODO
    source_space = source.function_space()
    if isinstance(target_space, firedrake.Function):
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


def mesh2mesh_project(source, target, adjoint=False, **kwargs):
    """
    Apply a mesh-to-mesh conservative projection to some
    ``source``, mapping to a ``target``.

    This function extends to the case of mixed spaces.

    Extra keyword arguments are passed to Firedrake's
    ``project`` function.

    :arg source: the :class:`Function` to be projected
    :arg target: the :class:`Function` which we
        seek to project onto
    :kwarg adjoint: apply the transposed projection
        operator?
    """
    if adjoint:
        return mesh2mesh_project_adjoint(source, target)
    source_space = source.function_space()
    assert isinstance(target, firedrake.Function)
    target_space = target.function_space()
    if source_space == target_space:
        target.assign(source)
    elif hasattr(target_space, 'num_sub_spaces'):
        assert hasattr(source_space, 'num_sub_spaces')
        assert target_space.num_sub_spaces() == source_space.num_sub_spaces()
        for s, t in zip(source.split(), target.split()):
            t.project(s, **kwargs)
    else:
        target.project(source, **kwargs)
    return target


def mesh2mesh_project_adjoint(target_b, source_b, **kwargs):
    """
    Apply the adjoint of a mesh-to-mesh conservative
    projection to some seed ``target_b``, mapping to
    ``source_b``.

    The notation used here is in terms of the adjoint of
    ``mesh2mesh_project``. However, this function may also
    be interpreted as a projector in its own right,
    mapping ``target_b`` to ``source_b``.

    Extra keyword arguments are passed to Firedrake's
    ``project`` function.

    :arg target_b: seed :class:`Function` from the target
        space of the forward projection
    :arg source_b: the :class:`Function` from the source
        space of the forward projection
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    target_space = target_b.function_space()
    assert isinstance(source_b, firedrake.Function)
    source_space = source_b.function_space()

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
