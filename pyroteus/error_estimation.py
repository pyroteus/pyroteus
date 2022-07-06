"""
Tools to automate goal-oriented error estimation.
"""
import firedrake
from firedrake.petsc import PETSc
import ufl


__all__ = ["get_dwr_indicator"]


@PETSc.Log.EventDecorator("pyroteus.form2indicator")
def form2indicator(F):
    """
    Multiply throughout in a form and
    assemble as a cellwise error
    indicator.

    :arg F: the form
    """
    mesh = F.ufl_domain()
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    p0test = firedrake.TestFunction(P0)
    indicator = firedrake.Function(P0)

    # Contributions from surface integrals
    flux_terms = 0
    integrals = F.integrals_by_type("exterior_facet")
    if len(integrals) > 0:
        for integral in integrals:
            ds = firedrake.ds(integral.subdomain_id())
            flux_terms += p0test * integral.integrand() * ds
    integrals = F.integrals_by_type("interior_facet")
    if len(integrals) > 0:
        for integral in integrals:
            dS = firedrake.dS(integral.subdomain_id())
            flux_terms += p0test("+") * integral.integrand() * dS
            flux_terms += p0test("-") * integral.integrand() * dS
    if flux_terms != 0:
        dx = firedrake.dx
        mass_term = firedrake.TrialFunction(P0) * p0test * dx
        sp = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        firedrake.solve(mass_term == flux_terms, indicator, solver_parameters=sp)

    # Contributions from volume integrals
    cell_terms = 0
    integrals = F.integrals_by_type("cell")
    if len(integrals) > 0:
        for integral in integrals:
            dx = firedrake.dx(integral.subdomain_id())
            cell_terms += p0test * integral.integrand() * dx
    indicator += firedrake.assemble(cell_terms)

    return indicator


@PETSc.Log.EventDecorator("pyroteus.indicator2estimator")
def indicator2estimator(indicator, absolute_value=False):
    """
    Deduce the error estimator value
    associated with a single error
    indicator field.

    :arg indicator: the error indicator
        :class:`firedrake.function.Function`
    :kwarg absolute_value: toggle
        whether to take the modulus
        on each element
    """
    if absolute_value:
        indicator.interpolate(abs(indicator))
    return indicator.vector().gather().sum()


def indicators2estimator(indicators, time_partition, **kwargs):
    r"""
    Deduce the error estimator value
    associated with error indicator
    fields defined over a :class:`~.MeshSeq`.

    :arg indicators: the list of list of
        error indicator
        :class:`firedrake.function.Function`\s
    :arg time_partition: the
        :class:`~.TimePartition` instance for
        the problem being solved
    """
    estimator = 0
    for by_mesh, dt in zip(indicators, time_partition.timesteps):
        time_integrated = sum([indicator2estimator(indi, **kwargs) for indi in by_mesh])
        estimator += dt * time_integrated
    return estimator


@PETSc.Log.EventDecorator("pyroteus.form2estimator")
def form2estimator(F, **kwargs):
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
    return indicator2estimator(indicator, **kwargs)


@PETSc.Log.EventDecorator("pyroteus.get_dwr_indicator")
def get_dwr_indicator(F, adjoint_error, test_space=None):
    """
    Generate a dual weighted residual (DWR)
    error indicator, given a form and an
    approximation of the error in the adjoint
    solution.

    :arg F: the form
    :arg adjoint_error: the approximation to
        the adjoint error, either as a single
        :class:`firedrake.function.Function`,
        or in a dictionary
    :kwarg test_space: the
        :class:`firedrake.functionspaceimpl.FunctionSpace`
        that the test function lives in, or an
        appropriate dictionary
    """
    mapping = {}
    if isinstance(adjoint_error, firedrake.Function):
        fs = test_space or adjoint_error.function_space()
        if F.ufl_domain() != fs.mesh():
            raise ValueError(
                "Meshes underlying the form and adjoint error do not match."
            )
        mapping[firedrake.TestFunction(fs)] = adjoint_error
    else:
        if test_space is None:
            test_space = {
                key: firedrake.TestFunction(err.function_space())
                for key, err in adjoint_error.items()
            }
        for key, err in adjoint_error.items():
            if F.ufl_domain() != test_space[key].mesh():
                raise ValueError(
                    "Meshes underlying the form and adjoint error do not match."
                )
            mapping[firedrake.TestFunction(test_space[key])] = err
    return form2indicator(ufl.replace(F, mapping))
