"""
Drivers for solving adjoint problems on sequences of meshes.
"""
import firedrake
from firedrake.petsc import PETSc
from firedrake_adjoint import pyadjoint
from .interpolation import project
from .mesh_seq import MeshSeq
from .options import GoalOrientedParameters
from .time_partition import TimePartition
from .utility import AttrDict, norm, Function, pyrint
from collections.abc import Callable
from functools import wraps
import numpy as np


__all__ = ["AdjointMeshSeq", "annotate_qoi"]


def annotate_qoi(get_qoi: Callable) -> Callable:
    """
    Decorator that ensures QoIs are annotated properly.

    Should be applied to the :meth:`~.AdjointMeshSeq.get_qoi` method.
    """

    @wraps(get_qoi)
    def wrap_get_qoi(mesh_seq, solution_map, i):
        qoi = get_qoi(mesh_seq, solution_map, i)

        # Count number of arguments
        num_kwargs = 0 if qoi.__defaults__ is None else len(qoi.__defaults__)
        num_args = qoi.__code__.co_argcount - num_kwargs
        if num_args == 0:
            if mesh_seq.qoi_type not in ["end_time", "steady"]:
                raise ValueError(
                    "Expected qoi_type to be 'end_time' or 'steady',"
                    f" not '{mesh_seq.qoi_type}'."
                )
        elif num_args == 1:
            if mesh_seq.qoi_type != "time_integrated":
                raise ValueError(
                    "Expected qoi_type to be 'time_integrated',"
                    f" not '{mesh_seq.qoi_type}'."
                )
        else:
            raise ValueError(f"QoI should have 0 or 1 args, not {num_args}.")

        @PETSc.Log.EventDecorator("goalie.AdjointMeshSeq.evaluate_qoi")
        @wraps(qoi)
        def wrap_qoi(*args, **kwargs):
            j = firedrake.assemble(qoi(*args, **kwargs))
            if pyadjoint.tape.annotate_tape():
                j.block_variable.adj_value = 1.0
            return j

        mesh_seq.qoi = wrap_qoi
        return wrap_qoi

    return wrap_get_qoi


class AdjointMeshSeq(MeshSeq):
    """
    An extension of :class:`~.MeshSeq` to account for solving adjoint problems on a
    sequence of meshes.

    For time-dependent quantities of interest, the solver should access and modify
    :attr:`~AdjointMeshSeq.J`, which holds the QoI value.
    """

    def __init__(self, time_partition: TimePartition, initial_meshes: list, **kwargs):
        """
        :kwarg get_qoi: a function, with two arguments, a :class:`~.AdjointMeshSeq`,
            which returns a function of either one or two variables, corresponding to
            either an end time or time integrated quantity of interest, respectively,
            as well as an index for the :class:`~.MeshSeq`
        """
        if kwargs.get("parameters") is None:
            kwargs["parameters"] = GoalOrientedParameters()
        self.qoi_type = kwargs.pop("qoi_type")
        if self.qoi_type not in ["end_time", "time_integrated", "steady"]:
            raise ValueError(
                f"QoI type '{self.qoi_type}' not recognised."
                " Choose from 'end_time', 'time_integrated', or 'steady'."
            )
        super().__init__(time_partition, initial_meshes, **kwargs)
        if self.qoi_type == "steady" and not self.steady:
            raise ValueError(
                "QoI type is set to 'steady' but the time partition is not steady."
            )
        elif self.qoi_type != "steady" and self.steady:
            raise ValueError(
                f"Time partition is steady but the QoI type is set to '{self.qoi_type}'."
            )
        self._get_qoi = kwargs.get("get_qoi")
        self.J = 0
        self.controls = None
        self.qoi_values = []

    @property
    @pyadjoint.no_annotations
    def initial_condition(self):
        return super().initial_condition

    @annotate_qoi
    def get_qoi(self, solution_map: dict, i: int) -> Callable:
        if self._get_qoi is None:
            raise NotImplementedError("'get_qoi' is not implemented.")
        return self._get_qoi(self, solution_map, i)

    @pyadjoint.no_annotations
    @PETSc.Log.EventDecorator()
    def get_checkpoints(
        self, solver_kwargs: dict = {}, run_final_subinterval: bool = False
    ) -> list:
        """
        Solve forward on the sequence of meshes, extracting checkpoints corresponding
        to the starting fields on each subinterval.

        The QoI is also evaluated.

        :kwarg solver_kwargs: additional keyword arguments which will be passed to the
            solver
        :kwarg run_final_subinterval: toggle whether to solve the PDE on the final
            subinterval
        """

        # In some cases we run over all subintervals to check the QoI that is computed
        if run_final_subinterval:
            self.J = 0

        # Generate the checkpoints as in MeshSeq
        checkpoints = super().get_checkpoints(
            solver_kwargs=solver_kwargs, run_final_subinterval=run_final_subinterval
        )

        # Account for end time QoI
        if self.qoi_type in ["end_time", "steady"] and run_final_subinterval:
            qoi = self.get_qoi(checkpoints[-1], len(self) - 1)
            self.J = qoi(**solver_kwargs.get("qoi_kwargs", {}))
        return checkpoints

    @PETSc.Log.EventDecorator()
    def get_solve_blocks(
        self, field: str, subinterval: int, has_adj_sol: bool = True
    ) -> list:
        """
        Get all blocks of the tape corresponding to solve steps for prognostic solution
        field on a given subinterval.

        :arg field: name of the prognostic solution field
        :arg subinterval: subinterval index
        :kwarg has_adj_sol: if ``True``, only blocks with ``adj_sol`` attributes will be
            considered
        """
        solve_blocks = super().get_solve_blocks(field, subinterval)
        if not has_adj_sol:
            return solve_blocks

        # Check that adjoint solutions exist
        if all(block.adj_sol is None for block in solve_blocks):
            self.warning(
                "No block has an adjoint solution. Has the adjoint equation been solved?"
            )

        # Default adjoint solution to zero, rather than None
        for block in solve_blocks:
            if block.adj_sol is None:
                block.adj_sol = Function(
                    self.function_spaces[field][subinterval], name=field
                )
        return solve_blocks

    @PETSc.Log.EventDecorator()
    def solve_adjoint(
        self,
        solver_kwargs: dict = {},
        adj_solver_kwargs: dict = {},
        get_adj_values: bool = False,
        test_checkpoint_qoi: bool = False,
    ) -> dict:
        """
        Solve an adjoint problem on a sequence of subintervals.

        As well as the quantity of interest value, a dictionary
        of solution fields is computed, the contents of which
        give values at all exported timesteps, indexed first by
        the field label and then by type. The contents of these
        nested dictionaries are lists which are indexed first by
        subinterval and then by export. For a given exported
        timestep, the solution types are:

        * ``'forward'``: the forward solution after taking the
            timestep;
        * ``'forward_old'``: the forward solution before taking
            the timestep
        * ``'adjoint'``: the adjoint solution after taking the
            timestep;
        * ``'adjoint_next'``: the adjoint solution before taking
            the timestep (backwards).

        :kwarg solver_kwargs: a dictionary providing parameters
            to the solver. Any keyword arguments for the QoI
            should be included as a subdict with label 'qoi_kwargs'
        :kwarg adj_solver_kwargs: a dictionary providing parameters
            to the adjoint solver.
        :kwarg get_adj_values: additionally output adjoint
            actions at exported timesteps
        :kwarg test_checkpoint_qoi: solve over the final
            subinterval when checkpointing so that the QoI
            value can be checked across runs

        :return solution: an :class:`~.AttrDict` containing
            solution fields and their lagged versions.
        """
        num_subintervals = len(self)
        function_spaces = self.function_spaces
        P = self.time_partition
        solver = self.solver
        qoi_kwargs = solver_kwargs.get("qoi_kwargs", {})

        # Solve forward to get checkpoints and evaluate QoI
        checkpoints = self.get_checkpoints(
            solver_kwargs=solver_kwargs,
            run_final_subinterval=test_checkpoint_qoi,
        )
        J_chk = float(self.J)
        if test_checkpoint_qoi and np.isclose(J_chk, 0.0):
            self.warning("Zero QoI. Is it implemented as intended?")

        # Reset the QoI to zero
        self.J = 0

        # Create arrays to hold exported forward and adjoint solutions and their lagged
        # counterparts, as well as the adjoint actions, if requested
        labels = ("forward", "forward_old", "adjoint")
        if not self.steady:
            labels += ("adjoint_next",)
        if get_adj_values:
            labels += ("adj_value",)
        solutions = AttrDict(
            {
                field: AttrDict(
                    {
                        label: [
                            [
                                Function(fs, name=f"{field}_{label}")
                                for j in range(P.num_exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(function_spaces[field])
                        ]
                        for label in labels
                    }
                )
                for field in self.fields
            }
        )

        @PETSc.Log.EventDecorator("goalie.AdjointMeshSeq.solve_adjoint.evaluate_fwd")
        @wraps(solver)
        def wrapped_solver(subinterval, ic, **kwargs):
            """
            Decorator to allow the solver to stash its initial conditions as controls.

            :arg subinterval: the subinterval index
            :arg ic: the dictionary of initial condition :class:`~.Functions`

            All keyword arguments are passed to the solver.
            """
            init = AttrDict(
                {field: ic[field].copy(deepcopy=True) for field in self.fields}
            )
            self.controls = [pyadjoint.Control(init[field]) for field in self.fields]
            return solver(subinterval, init, **kwargs)

        # Clear tape
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()

        # Loop over subintervals in reverse
        seeds = None
        for i in reversed(range(num_subintervals)):
            stride = P.num_timesteps_per_export[i]
            num_exports = P.num_exports_per_subinterval[i]

            # Annotate tape on current subinterval
            checkpoint = wrapped_solver(i, checkpoints[i], **solver_kwargs)

            # Get seed vector for reverse propagation
            if i == num_subintervals - 1:
                if self.qoi_type in ["end_time", "steady"]:
                    qoi = self.get_qoi(checkpoint, i)
                    self.J = qoi(**qoi_kwargs)
                    if np.isclose(float(self.J), 0.0):
                        self.warning("Zero QoI. Is it implemented as intended?")
            else:
                with pyadjoint.stop_annotating():
                    for field, fs in function_spaces.items():
                        checkpoint[field].block_variable.adj_value = project(
                            seeds[field], fs[i], adjoint=True
                        )

            # Update adjoint solver kwargs
            for field in self.fields:
                for block in self.get_solve_blocks(field, i, has_adj_sol=False):
                    block.adj_kwargs.update(adj_solver_kwargs)

            # Solve adjoint problem
            with PETSc.Log.Event("goalie.AdjointMeshSeq.solve_adjoint.evaluate_adj"):
                m = pyadjoint.enlisting.Enlist(self.controls)
                with pyadjoint.stop_annotating():
                    with tape.marked_nodes(m):
                        tape.evaluate_adj(markings=True)

            # Loop over prognostic variables
            for field, fs in function_spaces.items():
                # Get solve blocks
                solve_blocks = self.get_solve_blocks(field, i)
                num_solve_blocks = len(solve_blocks)
                if num_solve_blocks == 0:
                    raise ValueError(
                        "Looks like no solves were written to tape!"
                        " Does the solution depend on the initial condition?"
                    )
                if fs[0].ufl_element() != solve_blocks[0].function_space.ufl_element():
                    raise ValueError(
                        f"Solve block list for field '{field}' contains mismatching"
                        f" finite elements: ({fs[0].ufl_element()} vs. "
                        f" {solve_blocks[0].function_space.ufl_element()})"
                    )

                # Detect whether we have a steady problem
                steady = self.steady or num_subintervals == num_solve_blocks == 1
                if steady and "adjoint_next" in checkpoint:
                    checkpoint.pop("adjoint_next")

                # Check that there are as many solve blocks as expected
                if len(solve_blocks[::stride]) >= num_exports:
                    self.warning(
                        "More solve blocks than expected:"
                        f" ({len(solve_blocks[::stride])} > {num_exports-1})."
                    )

                # Update forward and adjoint solution data based on block dependencies
                # and outputs
                sols = solutions[field]
                for j, block in zip(range(num_exports - 1), solve_blocks[::stride]):
                    # Current forward solution is determined from outputs
                    out = self._output(field, i, block)
                    if out is not None:
                        sols.forward[i][j].assign(out.saved_output)

                    # Current adjoint solution is determined from the adj_sol attribute
                    if block.adj_sol is not None:
                        sols.adjoint[i][j].assign(block.adj_sol)

                    # Lagged forward solution comes from dependencies
                    dep = self._dependency(field, i, block)
                    if dep is not None:
                        sols.forward_old[i][j].assign(dep.saved_output)

                    # Adjoint action also comes from dependencies
                    if get_adj_values and dep is not None:
                        sols.adj_value[i][j].assign(dep.adj_value.function)

                    # The adjoint solution at the 'next' timestep is determined from the
                    # adj_sol attribute of the next solve block
                    if not steady:
                        if j * stride + 1 < num_solve_blocks:
                            if solve_blocks[j * stride + 1].adj_sol is not None:
                                sols.adjoint_next[i][j].assign(
                                    solve_blocks[j * stride + 1].adj_sol
                                )
                        elif j * stride + 1 == num_solve_blocks:
                            if i + 1 < num_subintervals:
                                sols.adjoint_next[i][j].assign(
                                    project(
                                        sols.adjoint_next[i + 1][0], fs[i], adjoint=True
                                    )
                                )
                        else:
                            raise IndexError(
                                "Cannot extract solve block"
                                f" {j*stride+1} > {num_solve_blocks}."
                            )

                # Check non-zero adjoint solution/value
                if np.isclose(norm(solutions[field].adjoint[i][0]), 0.0):
                    self.warning(
                        f"Adjoint solution for field '{field}' on {self.th(i)}"
                        " subinterval is zero."
                    )
                if get_adj_values and np.isclose(norm(sols.adj_value[i][0]), 0.0):
                    self.warning(
                        f"Adjoint action for field '{field}' on {self.th(i)}"
                        " subinterval is zero."
                    )

            # Get adjoint action on each subinterval
            seeds = {
                field: Function(
                    function_spaces[field][i], val=control.block_variable.adj_value
                )
                for field, control in zip(self.fields, self.controls)
            }
            for field, seed in seeds.items():
                if not self.steady and np.isclose(norm(seed), 0.0):
                    self.warning(
                        f"Adjoint action for field '{field}' on {self.th(i)}"
                        " subinterval is zero."
                    )

            # Clear the tape to reduce the memory footprint
            tape.clear_tape()

        # Check the QoI value agrees with that due to the checkpointing run
        if self.qoi_type == "time_integrated" and test_checkpoint_qoi:
            if not np.isclose(J_chk, self.J):
                raise ValueError(
                    "QoI values computed during checkpointing and annotated"
                    f" run do not match ({J_chk} vs. {self.J})"
                )
        return solutions

    @staticmethod
    def th(num: int) -> str:
        """
        Convert from cardinal to ordinal.
        """
        end = int(str(num)[-1])
        try:
            c = {1: "st", 2: "nd", 3: "rd"}[end]
        except KeyError:
            c = "th"
        return f"{num}{c}"

    def _subintervals_not_checked(self):
        num_not_checked = len(self.check_convergence[not self.check_convergence])
        return self.check_convergence.argsort()[num_not_checked]

    def check_qoi_convergence(self):
        """
        Check for convergence of the fixed point iteration due to the relative
        difference in QoI value being smaller than the specified tolerance.

        The :attr:`AdjointMeshSeq.converged` attribute is set to ``True`` across all
        entries if convergence is detected.
        """
        if not self.check_convergence.any():
            self.info(
                "Skipping QoI convergence check because check_convergence contains"
                f" False values for indices {self._subintervals_not_checked}."
            )
            return self.converged
        if len(self.qoi_values) >= max(2, self.params.miniter + 1):
            qoi_, qoi = self.qoi_values[-2:]
            if abs(qoi - qoi_) < self.params.qoi_rtol * abs(qoi_):
                self.converged[:] = True
                pyrint(
                    f"QoI converged after {self.fp_iteration+1} iterations"
                    f" under relative tolerance {self.params.qoi_rtol}."
                )
        return self.converged
