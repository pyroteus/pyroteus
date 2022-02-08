"""
Drivers for goal-oriented error estimation on sequences of meshes.
"""
from .adjoint import AdjointMeshSeq
from firedrake import FunctionSpace, MeshHierarchy
from firedrake.petsc import PETSc


__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`AdjointMeshSeq` to account for
    goal-oriented problems.
    """

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.global_enrichment")
    def global_enrichment(
        self, enrichment_method="p", num_enrichments_h=1, num_enrichments_p=1, **kwargs
    ):
        """
        Solve the forward and adjoint problems
        associated with :attr:`solver` in a
        sequence of globally enriched spaces.

        Currently, global enrichment may be
        achieved using one of:
        * h-refinement (``enrichment_method = 'h'``);
        * p-refinement (``enrichment_method = 'p'``);
        * hp-refinement (``enrichment_method = 'hp'``).

        The number of refinements in each direction
        may be controlled by the keyword arguments
        ``num_enrichments_h`` and ``num_enrichments_p``.

        Any additional keyword arguments are passed
        to the :attr:`solve_adjoint` method of the
        enriched :class:`AdjointMeshSeq`.
        """
        assert enrichment_method in ("h", "p", "hp")
        assert num_enrichments_h >= 0
        assert num_enrichments_p >= 0
        assert num_enrichments_h > 0 or num_enrichments_p > 0

        # Apply h-refinement
        if "h" in enrichment_method and num_enrichments_h > 0:
            meshes = [
                MeshHierarchy(mesh, num_enrichments_h)[-1] for mesh in self.meshes
            ]
        else:
            meshes = self.meshes

        def get_function_spaces(mesh):
            """
            Apply p-refinement, if requested.
            """
            if num_enrichments_p == 0:
                return self._get_function_spaces(mesh)
            enriched_spaces = {}
            for label, fs in self.function_spaces.items():
                element = fs[0].ufl_element()
                if "p" in enrichment_method:
                    element = element.reconstruct(
                        degree=element.degree() + num_enrichments_p
                    )
                enriched_spaces[label] = FunctionSpace(mesh, element)
            return enriched_spaces

        # Solve adjoint in higher order space
        adj_mesh_seq = AdjointMeshSeq(
            self.time_partition,
            meshes,
            get_function_spaces,
            self._get_initial_condition,
            self._get_solver,
            self._get_qoi,
            qoi_type=self.qoi_type,
            steady=self.steady,
        )
        return adj_mesh_seq.solve_adjoint(**kwargs)

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.fixed_point_iteration")
    def fixed_point_iteration(self, **kwargs):
        """
        Apply goal-oriented mesh adaptation using
        a fixed point iteration.
        """
        raise NotImplementedError  # TODO
