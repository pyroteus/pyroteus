"""
Driver functions for derivative recovery.
"""
from __future__ import absolute_import
from .utility import *


__all__ = ["recover_hessian", "recover_boundary_hessian"]


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
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "gamg",
        "fieldsplit_1_mg_levels_ksp_max_it": 5,
    }
    if COMM_WORLD.size == 1:
        sp["fieldsplit_0_pc_type"] = "ilu"
        sp["fieldsplit_1_mg_levels_pc_type"] = "ilu"
    else:
        sp["fieldsplit_0_pc_type"] = "bjacobi"
        sp["fieldsplit_0_sub_ksp_type"] = "preonly"
        sp["fieldsplit_0_sub_pc_type"] = "ilu"
        sp["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
        sp["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
        sp["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"
    solve(a == L, l2_projection, solver_parameters=sp)
    return l2_projection.split()


def recover_boundary_hessian(f, mesh, method='L2', **kwargs):
    """
    Recover the Hessian of a scalar field
    on the domain boundary.

    :arg f: dictionary of boundary tags and corresponding
        fields, which we seek to recover
    :arg mesh: the mesh
    :kwarg method: recovery method
    """
    # from math import gram_schmidt  # TODO

    if method.upper() != 'L2':
        raise NotImplementedError
    d = mesh.topological_dimension()
    assert d in (2, 3)

    # Apply Gram-Schmidt to get tangent vectors
    n = FacetNormal(mesh)
    # s = gram_schmidt(n)
    s = [perp(n)]
    ns = as_vector([n, *s])

    # Setup
    P1 = FunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    boundary_tag = kwargs.get('boundary_tag', 'on_boundary')
    Hs, v = TrialFunction(P1), TestFunction(P1)
    l2_proj = [[Function(P1) for i in range(d-1)] for j in range(d-1)]
    h = interpolate(CellSize(mesh), FunctionSpace(mesh, "DG", 0))
    h = Constant(1/h.vector().gather().max()**2)

    # Arbitrary value on domain interior
    a = v*Hs*dx
    L = v*h*dx

    # Hessian on boundary
    nullspace = VectorSpaceBasis(constant=True)
    sp = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 20,
        "ksp_rtol": 1.0e-05,
        "pc_type": "sor",
    }
    for j, s1 in enumerate(s):
        for i, s0 in enumerate(s):
            bcs = []
            for tag, fi in f.items():
                a_bc = v*Hs*ds(tag)
                L_bc = -dot(s0, grad(v))*dot(s1, grad(fi))*ds(tag)
                bcs.append(EquationBC(a_bc == L_bc, l2_proj[i][j], tag))
            solve(a == L, l2_proj[i][j], bcs=bcs,
                  nullspace=nullspace, solver_parameters=sp)

    # Construct tensor field
    Hbar = Function(P1_ten)
    if d == 2:
        Hsub = abs(l2_proj[i][j])
        H = as_matrix([[h, 0],
                       [0, Hsub]])
    else:
        raise NotImplementedError  # TODO

    # Arbitrary value in domain interior
    sigma, tau = TrialFunction(P1_ten), TestFunction(P1_ten)
    a = inner(tau, sigma)*dx
    L = inner(tau, h*Identity(d))*dx

    # Boundary values imposed as in [Loseille et al. 2011]
    a_bc = inner(tau, sigma)*ds
    L_bc = inner(tau, dot(transpose(ns), dot(H, ns)))*ds
    bcs = EquationBC(a_bc == L_bc, Hbar, boundary_tag)
    solve(a == L, Hbar, bcs=bcs, solver_parameters=sp)
    return Hbar
