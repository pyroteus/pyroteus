from firedrake import *


__all__ = ["get_dwr_indicator"]


def form2indicator(F):
    """
    Multiply throughout in a form and
    assemble as a cellwise error
    indicator.

    :arg F: the form
    """
    mesh = F.ufl_domain()
    P0 = FunctionSpace(mesh, "DG", 0)
    p0test = TestFunction(P0)
    indicator = Function(P0)

    # Contributions from surface integrals
    flux_terms = 0
    integrals = F.integrals_by_type('exterior_facet')
    if len(integrals) > 0:
        for integral in integrals:
            flux_terms += p0test*integral.integrand()*ds(integral.subdomain_id())
    integrals = F.integrals_by_type('interior_facet')
    if len(integrals) > 0:
        for integral in integrals:
            flux_terms += p0test('+')*integral.integrand()*dS(integral.subdomain_id())
            flux_terms += p0test('-')*integral.integrand()*dS(integral.subdomain_id())
    if flux_terms != 0:
        mass_term = TrialFunction(P0)*p0test*dx
        sp = {
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
        }
        solve(mass_term == flux_terms, indicator, solver_parameters=sp)

    # Contributions from volume integrals
    cell_terms = 0
    integrals = F.integrals_by_type('cell')
    if len(integrals) > 0:
        for integral in integrals:
            cell_terms += p0test*integral.integrand()*dx(integral.subdomain_id())
    indicator += assemble(cell_terms)

    return indicator


def form2estimator(F, absolute_value=False):
    """
    Multiply throughout in a form,
    assemble as a cellwise error
    indicator and sum over all
    mesh elements.

    :arg F: the form
    :kwarg absolute_value: toggle
        whether to take the modulus
        on each element
    """
    indicator = form2indicator(F)
    if absolute_value:
        indicator.interpolate(abs(indicator))
    return indicator.vector().gather().sum()


def get_dwr_indicator(F, adjoint_error):
    """
    Generate a dual weighted residual (DWR)
    error indicator, given a form and an
    approximation of the error in the adjoint
    solution.

    :arg F: the form
    :arg adjoint_error: the approximation to
        the adjoint error
    """
    fs = adjoint_error.function_space()
    if F.ufl_domain() != fs.mesh():
        raise ValueError("Meshes underlying the form and adjoint error do not match.")
    return form2indicator(replace(F, {TestFunction(fs): adjoint_error}))
