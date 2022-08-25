"""
Drivers for goal-oriented error estimation on sequences of meshes.
"""
from .adjoint import AdjointMeshSeq
from .error_estimation import get_dwr_indicator, indicators2estimator
from .log import pyrint
from .mesh_seq import AdaptParameters
from .metric import MetricParameters
from firedrake import Function, FunctionSpace, MeshHierarchy, TransferManager, project
from firedrake.petsc import PETSc


__all__ = ["GoalOrientedParameters", "GoalOrientedMetricParameters", "GoalOrientedMeshSeq"]


class GoalOrientedParameters(AdaptParameters):
    """
    A class for holding parameters associated with
    goal-oriented adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters={}):
        self["qoi_rtol"] = 0.001  # Relative tolerance for QoI
        self["estimator_rtol"] = 0.001  # Relative tolerance for estimator

        super().__init__(parameters=parameters)


class GoalOrientedMetricParameters(GoalOrientedParameters):
    """
    A class for holding parameters associated with
    metric-based, goal-oriented adaptive mesh fixed
    point iteration loops.
    """

    def __init__(self, parameters={}):
        MetricParameters.__init__(self)
        super().__init__(parameters=parameters)


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`~.AdjointMeshSeq` to account for
    goal-oriented problems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_values = []

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.get_enriched_mesh_seq")
    def get_enriched_mesh_seq(self, enrichment_method="p", num_enrichments=1):
        """
        Solve the forward and adjoint problems
        associated with
        :meth:`~.GoalOrientedMeshSeq.solver`
        in a sequence of globally enriched spaces.

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
        associated with
        :meth:`~.GoalOrientedMeshSeq.solver` in a
        sequence of globally enriched spaces.

        Currently, global enrichment may be
        achieved using one of:
        * h-refinement (``enrichment_method = 'h'``);
        * p-refinement (``enrichment_method = 'p'``).

        The number of refinements may be controlled by
        the keyword argument ``num_enrichments``.

        :kwarg kwargs: keyword arguments to pass to the
            :meth:`~.AdjointMeshSeq.solve_adjoint` method
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
                indi_e = indicator_fn(F, u_star_e)

                # Project back to the base space
                indi = project(indi_e, P0)
                indi.interpolate(abs(indi))
                indicator.append(indi)
            indicators.append(indicator)
        return sols, indicators

    def check_estimator_convergence(self):
        """
        Check for convergence of the fixed point iteration
        due to the relative difference in error estimator
        value being smaller than the specified tolerance.

        :return: ``True`` if converged, else ``False``
        """
        P = self.params
        self.converged = False
        if len(self.estimator_values) < max(2, P.miniter):
            return
        self.converged = True
        ee_ = self.estimator_values[-2]
        ee = self.estimator_values[-1]
        if abs(ee - ee_) < P.estimator_rtol * abs(ee_):
            self.converged = True

    @PETSc.Log.EventDecorator("pyroteus.GoalOrientedMeshSeq.fixed_point_iteration")
    def fixed_point_iteration(
        self,
        adaptor,
        update_params=None,
        enrichment_kwargs={},
        adj_kwargs={},
        indicator_fn=get_dwr_indicator,
    ):
        r"""
        Apply goal-oriented mesh adaptation using
        a fixed point iteration loop.

        :arg adaptor: function for adapting the mesh sequence.
            Its arguments are the :class:`~.MeshSeq` instance, the
            dictionary of solution
            :class:`firedrake.function.Function`\s and the
            list of error indicators
        :kwarg update_params: function for updating
            :attr:`~.GoalOrientedMeshSeq.params` at each
            iteration. Its arguments are the parameter class and
            the fixed point iteration
        :kwarg enrichment_kwargs: keyword arguments to pass
            to the global enrichment method
        :kwarg adj_kwargs: keyword arguments to pass to the
            adjoint solver
        :kwarg indicator_fn: function for error indication,
            which takes the form, adjoint error and enriched
            space(s) as arguments
        """
        P = self.params
        self.element_counts = [self.count_elements()]
        self.qoi_values = []
        self.estimator_values = []
        self.converged = False
        msg = "Terminated due to {:s} convergence after {:d} iterations"
        for fp_iteration in range(P.maxiter):
            if update_params is not None:
                update_params(P, fp_iteration)

            # Indicate errors over all meshes
            sols, indicators = self.indicate_errors(
                enrichment_kwargs=enrichment_kwargs,
                adj_kwargs=adj_kwargs,
                indicator_fn=indicator_fn,
            )

            # Check for QoI convergence
            # TODO: Put this check inside the adjoint solve as
            #       an optional return condition so that we
            #       can avoid unnecessary extra solves
            self.qoi_values.append(self.J)
            self.check_qoi_convergence()
            if self.converged:
                pyrint(msg.format("QoI", fp_iteration + 1))
                break

            # Check for error estimator convergence
            ee = indicators2estimator(indicators, self.time_partition)
            self.estimator_values.append(ee)
            self.check_estimator_convergence()
            if self.converged:
                pyrint(msg.format("error estimator", fp_iteration + 1))
                break

            # Adapt meshes and log element counts
            adaptor(self, sols, indicators)
            self.element_counts.append(self.count_elements())

            # Check for element count convergence
            self.check_element_count_convergence()
            if self.converged:
                pyrint(msg.format("element count", fp_iteration + 1))
                break
        if not self.converged:
            pyrint(f"Failed to converge in {P.maxiter} iterations")
