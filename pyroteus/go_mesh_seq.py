"""
Drivers for goal-oriented error estimation on sequences of meshes.
"""
from .adjoint import AdjointMeshSeq
from .error_estimation import get_dwr_indicator
from firedrake import Function, FunctionSpace, MeshHierarchy, TransferManager, project
from firedrake.petsc import PETSc


__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`AdjointMeshSeq` to account for
    goal-oriented problems.
    """

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.get_enriched_mesh_seq")
    def get_enriched_mesh_seq(self, enrichment_method="p", num_enrichments=1):
        """
        Solve the forward and adjoint problems
        associated with :attr:`solver` in a
        sequence of globally enriched spaces.

        Currently, global enrichment may be
        achieved using one of:
        * h-refinement (``enrichment_method = 'h'``);
        * p-refinement (``enrichment_method = 'p'``).

        The number of refinements may be controlled by
        the keyword argument ``num_enrichments``.
        """
        if enrichment_method not in ("h", "p"):
            raise ValueError(f"Enrichment method {enrichment_method} not supported")
        if num_enrichments <= 0:
            raise ValueError("A positive number of enrichments is required")

        # Apply h-refinement
        if enrichment_method == "h":
            meshes = [MeshHierarchy(mesh, num_enrichments)[-1] for mesh in self.meshes]
        else:
            meshes = self.meshes

        def get_function_spaces(mesh):
            """
            Apply p-refinement, if requested.
            """
            if enrichment_method == "h":
                return self._get_function_spaces(mesh)
            enriched_spaces = {}
            for label, fs in self.function_spaces.items():
                element = fs[0].ufl_element()
                element = element.reconstruct(degree=element.degree() + num_enrichments)
                enriched_spaces[label] = FunctionSpace(mesh, element)
            return enriched_spaces

        # Construct enriched AdjointMeshSeq
        return AdjointMeshSeq(
            self.time_partition,
            meshes,
            get_function_spaces=get_function_spaces,
            get_initial_condition=self._get_initial_condition,
            get_form=self._get_form,
            get_solver=self._get_solver,
            get_qoi=self._get_qoi,
            get_bcs=self._get_bcs,
            qoi_type=self.qoi_type,
        )

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.global_enrichment")
    def global_enrichment(self, enrichment_method="p", num_enrichments=1, **kwargs):
        """
        Solve the forward and adjoint problems
        associated with :attr:`solver` in a
        sequence of globally enriched spaces.

        Currently, global enrichment may be
        achieved using one of:
        * h-refinement (``enrichment_method = 'h'``);
        * p-refinement (``enrichment_method = 'p'``).

        The number of refinements may be controlled by
        the keyword argument ``num_enrichments``.

        :kwarg kwargs: keyword arguments to pass to the
            :attr:`solve_adjoint` method of the
            enriched :class:`AdjointMeshSeq`.
        """
        mesh_seq = self.get_enriched_mesh_seq(
            enrichment_method=enrichment_method,
            num_enrichments=num_enrichments,
        )
        return mesh_seq.solve_adjoint(**kwargs)

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.indicate_errors")
    def indicate_errors(
        self, enrichment_kwargs={}, adj_kwargs={}, indicator_fn=get_dwr_indicator
    ):
        """
        Compute goal-oriented error indicators for each
        subinterval based on solving the adjoint problem
        in a globally enriched space.

        :kwarg enrichment_kwargs: keyword arguments to pass
            to the global enrichment method
        :kwarg adj_kwargs: keyword arguments to pass to the
            adjoint solver
        :kwarg indicator_fn: function for error indication,
            which takes the form, adjoint error and enriched
            space(s) as arguments
        """
        enrichment_method = enrichment_kwargs.get("enrichment_method", "p")
        if enrichment_method == "h":
            tm = TransferManager()
            transfer = tm.prolong
        else:

            def transfer(source, target):
                target.interpolate(source)

        mesh_seq_e = self.get_enriched_mesh_seq(**enrichment_kwargs)
        sols = self.solve_adjoint(**adj_kwargs)
        sols_e = mesh_seq_e.solve_adjoint(**adj_kwargs)
        indicators = []
        FWD, ADJ = "forward", "adjoint"
        FWD_OLD = "forward" if self.steady else "forward_old"
        ADJ_NEXT = "adjoint" if self.steady else "adjoint_next"
        for i, mesh in enumerate(self):
            P0 = FunctionSpace(self[i], "DG", 0)
            indicator = []

            # Get Functions
            u, u_, u_star, u_star_next, u_star_e = {}, {}, {}, {}, {}
            solutions = {}
            enriched_spaces = {f: mesh_seq_e.function_spaces[f][i] for f in self.fields}
            mapping = {}
            for f, fs_e in enriched_spaces.items():
                u[f] = Function(fs_e)
                u_[f] = Function(fs_e)
                mapping[f] = (u[f], u_[f])
                u_star[f] = Function(fs_e)
                u_star_next[f] = Function(fs_e)
                u_star_e[f] = Function(fs_e)
                solutions[f] = [
                    sols[f][FWD][i],
                    sols[f][FWD_OLD][i],
                    sols[f][ADJ][i],
                    sols[f][ADJ_NEXT][i],
                    sols_e[f][ADJ][i],
                    sols_e[f][ADJ_NEXT][i],
                ]

            # Get form in enriched space
            F = mesh_seq_e.form(i, mapping)

            for j in range(len(sols[self.fields[0]]["forward"][i])):
                for f in self.fields:

                    # Update fields
                    transfer(sols[f][FWD][i][j], u[f])
                    transfer(sols[f][FWD_OLD][i][j], u_[f])
                    transfer(sols[f][ADJ][i][j], u_star[f])
                    transfer(sols[f][ADJ_NEXT][i][j], u_star_next[f])

                    # Combine adjoint solutions as appropriate
                    u_star[f].assign(0.5 * (u_star[f] + u_star_next[f]))
                    u_star_e[f].assign(
                        0.5 * (sols_e[f][ADJ][i][j] + sols_e[f][ADJ_NEXT][i][j])
                    )
                    u_star_e[f] -= u_star[f]

                # Evaluate error indicator
                indi_e = indicator_fn(F, u_star_e, enriched_spaces)

                # Project back to the base space
                indi = project(indi_e, P0)
                indi.interpolate(abs(indi))
                indicator.append(indi)
            indicators.append(indicator)
        return sols, indicators

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.fixed_point_iteration")
    def fixed_point_iteration(self, **kwargs):
        """
        Apply goal-oriented mesh adaptation using
        a fixed point iteration.
        """
        raise NotImplementedError  # TODO
