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

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.get_enriched_mesh_seq")
    def get_enriched_mesh_seq(self, enrichment_method="p", num_h_enrichments=1, num_p_enrichments=1):
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
        ``num_h_enrichments`` and ``num_p_enrichments``.
        """
        assert enrichment_method in ("h", "p", "hp")
        assert num_h_enrichments >= 0
        assert num_p_enrichments >= 0
        assert num_h_enrichments > 0 or num_p_enrichments > 0

        # Apply h-refinement
        if "h" in enrichment_method and num_h_enrichments > 0:
            meshes = [
                MeshHierarchy(mesh, num_h_enrichments)[-1] for mesh in self.meshes
            ]
        else:
            meshes = self.meshes

        def get_function_spaces(mesh):
            """
            Apply p-refinement, if requested.
            """
            if num_p_enrichments == 0:
                return self._get_function_spaces(mesh)
            enriched_spaces = {}
            for label, fs in self.function_spaces.items():
                element = fs[0].ufl_element()
                if "p" in enrichment_method:
                    element = element.reconstruct(
                        degree=element.degree() + num_p_enrichments
                    )
                enriched_spaces[label] = FunctionSpace(mesh, element)
            return enriched_spaces

        # Construct enriched AdjointMeshSeq
        return AdjointMeshSeq(
            self.time_partition,
            meshes,
            get_function_spaces,
            self._get_initial_condition,
            self._get_form,
            self._get_solver,
            self._get_qoi,
            qoi_type=self.qoi_type,
            steady=self.steady,
        )

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.global_enrichment")
    def global_enrichment(self, enrichment_method="p", num_h_enrichments=1, num_p_enrichments=1, **kwargs):
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
        ``num_h_enrichments`` and ``num_p_enrichments``.

        :kwarg kwargs: keyword arguments to pass to the
            :attr:`solve_adjoint` method of the
            enriched :class:`AdjointMeshSeq`.
        """
        mesh_seq = self.get_enriched_mesh_seq(
            enrichment_method=enrichment_method,
            num_h_enrichments=num_h_enrichments,
            num_p_enrichments=num_p_enrichments,
        )
        return mesh_seq.solve_adjoint(**kwargs)

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.fixed_point_iteration")
    def fixed_point_iteration(self, **kwargs):
        """
        Apply goal-oriented mesh adaptation using
        a fixed point iteration.
        """
        raise NotImplementedError  # TODO
