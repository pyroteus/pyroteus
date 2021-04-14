from __future__ import absolute_import
from .utility import *


__all__ = ["recover_hessian"]


def recover_hessian(f, method='L2', **kwargs):
    """
    Recover the Hessian of a scalar field.

    :arg f: the scalar field whose Hessian we seek to recover
    :kwarg method: recovery method
    """
    if method.upper() == 'L2':
        return double_l2_projection(f, **kwargs)[1]
    else:
        raise NotImplementedError


def double_l2_projection(f, mesh=None, target_spaces=None):
    r"""
    Recover the gradient and Hessian of a scalar field using a
    double :math:`L^2` projection.

    :arg f: the scalar field whose derivatives we seek to recover
    :kwarg mesh: the underlying mesh
    :kwarg target_spaces: the :class:`VectorFunctionSpace` and
        :class:`TensorFunctionSpace` the recovered gradient and
        Hessian should live in
    """
    mesh = mesh or f.function_space().mesh()
    if target_spaces is None:
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    else:
        P1_vec, P1_ten = target_spaces
    W = P1_vec*P1_ten
    g, H = TrialFunctions(W)
    phi, tau = TestFunctions(W)
    l2_projection = Function(W)
    n = FacetNormal(mesh)

    # The formulation is chosen such that f does not need to have any
    # finite element derivatives
    a = inner(tau, H)*dx + inner(div(tau), g)*dx - dot(g, dot(tau, n))*ds
    a += inner(phi, g)*dx
    L = f*dot(phi, n)*ds - f*div(phi)*dx

    # Apply stationary preconditioners in the Schur complement to get away
    # with applying GMRES to the whole mixed system
    sp = {
        "mat_type": "aij",
        "ksp_type": "gmres",
        "ksp_max_it": 20,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        "pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "gamg",
    }
    solve(a == L, l2_projection, solver_parameters=sp)
    return l2_projection.split()
