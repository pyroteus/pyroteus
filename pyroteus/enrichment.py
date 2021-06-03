from __future__ import absolute_import
from .utility import *


__all__ = ["global_enrichment"]


def global_enrichment(solver, initial_condition, qoi, function_spaces, time_partition, **kwargs):
    """
    Solve the forward and adjoint problems
    associated with some ``solver`` in a
    sequence of globally enriched spaces.

    Currently, global enrichment may be
    achieved using one of:
    * h-refinement (``enrichment_method = 'h'``);
    * p-refinement (``enrichment_method = 'p'``);
    * hp-refinement (``enrichment_method = 'hp'``).

    The number of refinements in each direction
    may be controlled by the keyword arguments
    ``num_enrichments_h`` and ``num_enrichments_p``.

    :kwarg solve_adjoint: optional adjoint solver
        which should have the same calling
        sequence as that found in pyroteus.adjoint
    """
    enriched_spaces = {}
    for label, function_space in function_spaces.items():
        solve_adjoint = kwargs.pop('solve_adjoint', None)
        if solve_adjoint is None:
            from .adjoint import solve_adjoint
        enrichment_method = kwargs.pop('enrichment_method', 'p')
        assert enrichment_method in ('h', 'p', 'hp')
        num_enrichments_h = kwargs.pop('num_enrichments_h', 1)
        num_enrichments_p = kwargs.pop('num_enrichments_p', 1)

        # Check consistency of FunctionSpaces
        element = function_space[0].ufl_element()
        for fs in function_space:
            assert element == fs.ufl_element(), "Finite elements are not identical"

        # Apply h-refinement
        meshes = [fs.mesh() for fs in function_space]
        if 'h' in enrichment_method:
            assert num_enrichments_h > 0
            meshes = [MeshHierarchy(mesh, num_enrichments_h)[-1] for mesh in meshes]

        # Apply p-refinement
        if 'p' in enrichment_method:
            assert num_enrichments_p > 0
            element = element.reconstruct(degree=element.degree() + num_enrichments_p)

        # Construct enriched space
        enriched_spaces[label] = [FunctionSpace(mesh, element) for mesh in meshes]

    # Solve adjoint in higher order space
    return solve_adjoint(solver, initial_condition, qoi, enriched_spaces, time_partition, **kwargs)


def effectivity_index(error_indicator, Je):
    """
    Overestimation factor of some error estimator
    for the QoI error.

    Note that this is only typically used for simple
    steady-state problems with analytical solutions.

    :arg error_indicator: a :math:`\mathbb P0`
        :class:`Function` which localises
        contributions to an error estimator to
        individual elements
    :arg Je: error in quantity of interest
    """
    assert isinstance(error_indicator, Function), "Error indicator must return a Function"
    el = error_indicator.ufl_element()
    assert (el.family(), el.degree()) == ('Discontinuous Lagrange', 0), "Error indicator must be P0"
    eta = error_indicator.vector().gather().sum()
    return np.abs(eta/Je)
