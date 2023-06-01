"""
Tools to automate goal-oriented error estimation.
"""
from .time_partition import TimePartition
import firedrake
from firedrake import Function, FunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
import ufl
from collections.abc import Iterable
from typing import Dict, Optional, Union


__all__ = ["get_dwr_indicator"]


@PETSc.Log.EventDecorator()
def form2indicator(F: ufl.form.Form) -> Function:
    r"""
    Given a 0-form, multiply the integrand of each of its integrals by a
    :math:`\mathbb P0` test function and reassemble to give an element-wise error
    indicator.

    Note that a 0-form does not contain any :class:`firedrake.ufl_expr.TestFunction`\s
    or :class:`firedrake.ufl_expr.TrialFunction`\s.

    :arg F: the 0-form
    :return: the corresponding error indicator field
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
def indicators2estimator(
    indicators: Iterable, time_partition: TimePartition, absolute_value: bool = False
) -> float:
    r"""
    Deduce the error estimator value associated with error indicator fields defined over
    a :class:`~.MeshSeq`.

    :arg indicators: the list of list of error indicator
        :class:`firedrake.function.Function`\s
    :arg time_partition: the :class:`~.TimePartition` instance for the problem being
        solved
    :kwarg absolute_value: toggle whether to take the modulus on each element
    """
    if not isinstance(indicators, dict):
        raise TypeError(
            f"Expected 'indicators' to be a dict, not '{type(indicators)}'."
        )
    if not isinstance(time_partition, TimePartition):
        raise TypeError(
            f"Expected 'time_partition' to be a TimePartition, not '{type(time_partition)}'."
        )
    if not isinstance(absolute_value, bool):
        raise TypeError(
            f"Expected 'absolute_value' to be a bool, not '{type(absolute_value)}'."
        )
    estimator = 0
    for field, by_field in indicators.items():
        if field not in time_partition.fields:
            raise ValueError(
                f"Key '{field}' does not exist in the TimePartition provided."
            )
        if isinstance(by_field, Function) or not isinstance(by_field, Iterable):
            raise TypeError(
                f"Expected values of 'indicators' to be iterables, not '{type(by_field)}'."
            )
        for by_mesh, dt in zip(by_field, time_partition.timesteps):
            if isinstance(by_mesh, Function) or not isinstance(by_mesh, Iterable):
                raise TypeError(
                    f"Expected entries of 'indicators' to be iterables, not '{type(by_mesh)}'."
                )
            for indicator in by_mesh:
                if absolute_value:
                    indicator.interpolate(abs(indicator))
                estimator += dt * indicator.vector().gather().sum()
    return estimator


@PETSc.Log.EventDecorator()
def get_dwr_indicator(
    F, adjoint_error: Function, test_space: Optional[Union[WithGeometry, Dict]] = None
) -> Function:
    r"""
    Given a 1-form and an approximation of the error in the adjoint solution, compute a
    dual weighted residual (DWR) error indicator.

    Note that each term of a 1-form contains only one
    :class:`firedrake.ufl_expr.TestFunction`. The 1-form most commonly corresponds to the
    variational form of a PDE. If the PDE is linear, it should be written as in the
    nonlinear case (i.e., with the solution field in place of any
    :class:`firedrake.ufl_expr.TrialFunction`\s.

    :arg F: the form
    :arg adjoint_error: the approximation to the adjoint error, either as a single
        :class:`firedrake.function.Function`, or in a dictionary
    :kwarg test_space: the
        :class:`firedrake.functionspaceimpl.WithGeometry` that the test function lives
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
