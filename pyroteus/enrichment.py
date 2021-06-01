from __future__ import absolute_import
from .utility import *


__all__ = ["global_enrichment"]


def global_enrichment(solver, initial_condition, function_spaces, time_partition, **kwargs):
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
        sequence as that found in .adjoint
    """
    solve_adjoint = kwargs.pop('solve_adjoint', None)
    if solve_adjoint is None:
        from .adjoint import solve_adjoint
    enrichment_method = kwargs.pop('enrichment_method', 'p')
    assert enrichment_method in ('h', 'p', 'hp')
    num_enrichments_h = kwargs.pop('num_enrichments_h', 1)
    num_enrichments_p = kwargs.pop('num_enrichments_p', 1)

    # Check consistency of FunctionSpaces
    element = function_spaces[0].ufl_element()
    for fs in function_spaces:
        assert element == fs.ufl_element(), "Finite elements are not identical"

    # Apply h-refinement
    meshes = [fs.mesh() for fs in function_spaces]
    if 'h' in enrichment_method:
        assert num_enrichments_h > 0
        meshes = [MeshHierarchy(mesh, num_enrichments_h)[-1] for mesh in meshes]

    # Apply p-refinement
    if 'p' in enrichment_method:
        assert num_enrichments_p > 0
        element = element.reconstruct(degree=element.degree() + num_enrichments_p)

    # Construct enriched space
    enriched_spaces = [FunctionSpace(mesh, element) for mesh in meshes]

    # Solve adjoint in higher order space
    return solve_adjoint(solver, initial_condition, enriched_spaces, time_partition, **kwargs)
