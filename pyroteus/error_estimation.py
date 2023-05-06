"""
Tools to automate goal-oriented error estimation.
"""
from .time_partition import TimePartition
import firedrake
from firedrake import Function, FunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
import ufl
from typing import Dict, Optional, Union


__all__ = ["get_dwr_indicator"]


@PETSc.Log.EventDecorator()
def form2indicator(F) -> Function:
    """
    Multiply throughout in a form and assemble as a cellwise error indicator.

    :arg F: the form
    """
    if not isinstance(F, ufl.form.Form):
        raise TypeError(f"Expected 'F' to be a Form, not '{type(F)}'.")
    mesh = F.ufl_domain()
    P0 = FunctionSpace(mesh, "DG", 0)
    p0test = firedrake.TestFunction(P0)
    indicator = Function(P0)

    # Contributions from surface integrals
    flux_terms = 0
    for integral in F.integrals_by_type("exterior_facet"):
        ds = firedrake.ds(integral.subdomain_id())
        flux_terms += p0test * integral.integrand() * ds
    for integral in F.integrals_by_type("interior_facet"):
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
    for integral in F.integrals_by_type("cell"):
        dx = firedrake.dx(integral.subdomain_id())
        cell_terms += p0test * integral.integrand() * dx
    if cell_terms != 0:
        indicator += firedrake.assemble(cell_terms)

    return indicator


@PETSc.Log.EventDecorator()
def indicator2estimator(indicator: Function, absolute_value: bool = False) -> float:
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


def indicators2estimator(
    indicators: list, time_partition: TimePartition, **kwargs
) -> float:
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


@PETSc.Log.EventDecorator()
def form2estimator(F, **kwargs) -> float:
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


@PETSc.Log.EventDecorator()
def get_dwr_indicator(
    F, adjoint_error: Function, test_space: Optional[Union[WithGeometry, Dict]] = None
) -> Function:
    """
    Generate a dual weighted residual (DWR) error indicator, given a form and an
    approximation of the error in the adjoint solution.

    :arg F: the form
    :arg adjoint_error: the approximation to the adjoint error, either as a single
        :class:`firedrake.function.Function`, or in a dictionary
    :kwarg test_space: the
        :class:`firedrake.functionspaceimpl.FunctionSpace` that the test function lives
        in, or an appropriate dictionary
    """
    mapping = {}
    if not isinstance(F, ufl.form.Form):
        raise TypeError(f"Expected 'F' to be a Form, not '{type(F)}'.")

    # Process input for adjoint_error as a dictionary
    if isinstance(adjoint_error, Function):
        name = adjoint_error.name()
        if test_space is None:
            test_space = {name: adjoint_error.function_space()}
        adjoint_error = {name: adjoint_error}
    elif not isinstance(adjoint_error, dict):
        raise TypeError(
            f"Expected 'adjoint_error' to be a Function or dict, not '{type(adjoint_error)}'."
        )

    # Process input for test_space as a dictionary
    if test_space is None:
        test_space = {key: err.function_space() for key, err in adjoint_error.items()}
    elif isinstance(test_space, WithGeometry):
        if len(adjoint_error.keys()) != 1:
            raise ValueError("Inconsistent input for 'adjoint_error' and 'test_space'.")
        test_space = {key: test_space for key in adjoint_error}
    elif not isinstance(test_space, dict):
        raise TypeError(
            f"Expected 'test_space' to be a FunctionSpace or dict, not '{type(test_space)}'."
        )

    # Construct the mapping for each component
    for key, err in adjoint_error.items():
        if key not in test_space:
            raise ValueError(f"Key '{key}' does not exist in the test space provided.")
        fs = test_space[key]
        if not isinstance(fs, WithGeometry):
            raise TypeError(
                f"Expected 'test_space['{key}']' to be a FunctionSpace, not '{type(fs)}'."
            )
        if F.ufl_domain() != err.function_space().mesh():
            raise ValueError(
                "Meshes underlying the form and adjoint error do not match."
            )
        if F.ufl_domain() != fs.mesh():
            raise ValueError("Meshes underlying the form and test space do not match.")
        mapping[firedrake.TestFunction(fs)] = err

    # Apply the mapping
    return form2indicator(ufl.replace(F, mapping))
