"""
Driver functions for derivative recovery.
"""
from __future__ import absolute_import
from .interpolation import clement_interpolant
from .utility import *
from petsc4py import PETSc as petsc4py


__all__ = ["recover_hessian", "recover_boundary_hessian"]


def recover_hessian(f, method='L2', **kwargs):
    """
    Recover the Hessian of a scalar field.

    :arg f: the scalar field whose Hessian we seek to recover
    :kwarg method: recovery method
    """
    if method.upper() == 'L2':
        g, H = double_l2_projection(f, **kwargs)
    elif method.capitalize() == 'Clement':
        mesh = kwargs.get('mesh') or f.function_space().mesh()
        g = clement_interpolant(firedrake.interpolate(ufl.grad(f),
                                firedrake.VectorFunctionSpace(mesh, "DG", 0)))
        H = clement_interpolant(firedrake.interpolate(ufl.grad(g),
                                firedrake.TensorFunctionSpace(mesh, "DG", 0)))
    elif method.upper() == 'ZZ':
        raise NotImplementedError("Zienkiewicz-Zhu recovery not yet implemented.")  # TODO
    else:
        raise ValueError(f"Recovery method '{method}' not recognised.")
    return H


@PETSc.Log.EventDecorator("pyroteus.double_l2_projection")
def double_l2_projection(f, mesh=None, target_spaces=None, mixed=False):
    r"""
    Recover the gradient and Hessian of a scalar field using a
    double :math:`L^2` projection.

    :arg f: the scalar field whose derivatives we seek to recover
    :kwarg mesh: the underlying mesh
    :kwarg target_spaces: the :class:`VectorFunctionSpace` and
        :class:`TensorFunctionSpace` the recovered gradient and
        Hessian should live in
    :kwarg mixed: solve as a mixed system, or separately?
    """
    mesh = mesh or f.function_space().mesh()
    if target_spaces is None:
        P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        P1_ten = firedrake.TensorFunctionSpace(mesh, "CG", 1)
    else:
        P1_vec, P1_ten = target_spaces
    if not mixed:
        g = firedrake.project(ufl.grad(f), P1_vec)
        H = firedrake.project(ufl.grad(g), P1_ten)
        return g, H
    W = P1_vec*P1_ten
    g, H = firedrake.TrialFunctions(W)
    phi, tau = firedrake.TestFunctions(W)
    l2_projection = firedrake.Function(W)
    n = ufl.FacetNormal(mesh)

    # The formulation is chosen such that f does not need to have any
    # finite element derivatives
    a = ufl.inner(tau, H)*ufl.dx \
        + ufl.inner(ufl.div(tau), g)*ufl.dx \
        - ufl.dot(g, ufl.dot(tau, n))*ufl.ds
    a += ufl.inner(phi, g)*ufl.dx
    L = f*ufl.dot(phi, n)*ufl.ds \
        - f*ufl.div(phi)*ufl.dx

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
    if firedrake.COMM_WORLD.size == 1:
        sp["fieldsplit_0_pc_type"] = "ilu"
        sp["fieldsplit_1_mg_levels_pc_type"] = "ilu"
    else:
        sp["fieldsplit_0_pc_type"] = "bjacobi"
        sp["fieldsplit_0_sub_ksp_type"] = "preonly"
        sp["fieldsplit_0_sub_pc_type"] = "ilu"
        sp["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
        sp["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
        sp["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"
    try:
        firedrake.solve(a == L, l2_projection, solver_parameters=sp)
    except firedrake.ConvergenceError:
        petsc4py.Sys.Print("L2 projection failed to converge with"
                           " iterative solver parameters, trying direct.")
        sp = {"pc_mat_factor_solver_type": "mumps"}
        firedrake.solve(a == L, l2_projection, solver_parameters=sp)
    return l2_projection.split()


@PETSc.Log.EventDecorator("pyroteus.recovery_boundary_hessian")
def recover_boundary_hessian(f, mesh, method='Clement', target_space=None, **kwargs):
    """
    Recover the Hessian of a scalar field
    on the domain boundary.

    :arg f: dictionary of boundary tags and corresponding
        fields, which we seek to recover, as well as an
        'interior' entry for the domain interior
    :arg mesh: the mesh
    :kwarg method: choose from 'L2' and 'Clement'
    :kwarg target_space: :class:`TensorFunctionSpace` in
        which the metric will exist
    """
    from pyroteus.math import construct_orthonormal_basis
    from pyroteus.metric import hessian_metric

    d = mesh.topological_dimension()
    assert d in (2, 3)

    # Apply Gram-Schmidt to get tangent vectors
    n = ufl.FacetNormal(mesh)
    s = construct_orthonormal_basis(n)
    ns = ufl.as_vector([n, *s])

    # Setup
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)
    P1_ten = target_space or firedrake.TensorFunctionSpace(mesh, "CG", 1)
    assert P1_ten.ufl_element().family() == 'Lagrange'
    assert P1_ten.ufl_element().degree() == 1
    boundary_tag = kwargs.get('boundary_tag', 'on_boundary')
    Hs = firedrake.TrialFunction(P1)
    v = firedrake.TestFunction(P1)
    l2_proj = [[firedrake.Function(P1) for i in range(d-1)] for j in range(d-1)]
    h = firedrake.interpolate(ufl.CellDiameter(mesh),
                              firedrake.FunctionSpace(mesh, "DG", 0))
    h = firedrake.Constant(1/h.vector().gather().max()**2)
    f.pop('interior')
    sp = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 20,
        "pc_type": "ilu",
    }

    if method.upper() == 'L2':

        # Arbitrary value on domain interior
        a = v*Hs*ufl.dx
        L = v*h*ufl.dx

        # Hessian on boundary
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        for j, s1 in enumerate(s):
            for i, s0 in enumerate(s):
                bcs = []
                for tag, fi in f.items():
                    a_bc = v*Hs*ufl.ds(tag)
                    L_bc = -ufl.dot(s0, ufl.grad(v))*ufl.dot(s1, ufl.grad(fi))*ufl.ds(tag)
                    bcs.append(firedrake.EquationBC(a_bc == L_bc, l2_proj[i][j], tag))
                firedrake.solve(a == L, l2_proj[i][j], bcs=bcs,
                                nullspace=nullspace, solver_parameters=sp)

    elif method.capitalize() == 'Clement':
        P0_vec = firedrake.VectorFunctionSpace(mesh, "DG", 0)
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        P1_vec = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        H = firedrake.Function(P1_ten)
        p0test = firedrake.TestFunction(P0_vec)
        p1test = firedrake.TestFunction(P1)
        fa = get_facet_areas(mesh)
        for tag, fi in f.items():
            source = firedrake.assemble(ufl.inner(p0test, ufl.grad(fi))/fa*ufl.ds)

            # Recover gradient
            c = clement_interpolant(source, boundary_tag=tag, target_space=P1_vec)

            # Recover Hessian
            H += clement_interpolant(firedrake.interpolate(ufl.grad(c), P0_ten),
                                     boundary_tag=tag, target_space=P1_ten)

        # Compute tangential components
        for j, s1 in enumerate(s):
            for i, s0 in enumerate(s):
                l2_proj[i][j] = firedrake.assemble(p1test*ufl.dot(ufl.dot(s0, H), s1)/fa*ufl.ds)
    else:
        raise ValueError(f"Recovery method '{method}' not supported"
                         " for Hessians on the boundary.")

    # Construct tensor field
    Hbar = firedrake.Function(P1_ten)
    if d == 2:
        Hsub = firedrake.interpolate(abs(l2_proj[0][0]), P1)
        H = ufl.as_matrix([[h, 0],
                           [0, Hsub]])
    else:
        Hsub = firedrake.Function(firedrake.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)))
        Hsub.interpolate(ufl.as_matrix([[l2_proj[0][0], l2_proj[0][1]],
                                        [l2_proj[1][0], l2_proj[1][1]]]))
        Hsub = hessian_metric(Hsub)
        H = ufl.as_matrix([[h, 0, 0],
                           [0, Hsub[0, 0], Hsub[0, 1]],
                           [0, Hsub[1, 0], Hsub[1, 1]]])

    # Arbitrary value on domain interior
    sigma = firedrake.TrialFunction(P1_ten)
    tau = firedrake.TestFunction(P1_ten)
    a = ufl.inner(tau, sigma)*ufl.dx
    L = ufl.inner(tau, h*ufl.Identity(d))*ufl.dx

    # Boundary values imposed as in [Loseille et al. 2011]
    a_bc = ufl.inner(tau, sigma)*ufl.ds
    L_bc = ufl.inner(tau, ufl.dot(ufl.transpose(ns), ufl.dot(H, ns)))*ufl.ds
    bcs = firedrake.EquationBC(a_bc == L_bc, Hbar, boundary_tag)
    firedrake.solve(a == L, Hbar, bcs=bcs, solver_parameters=sp)
    return Hbar
