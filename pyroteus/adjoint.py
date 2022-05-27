"""
Drivers for solving adjoint problems on sequences of meshes.
"""
import firedrake
from firedrake.petsc import PETSc
from firedrake_adjoint import pyadjoint
from .interpolation import project
from .mesh_seq import MeshSeq
from .utility import AttrDict, norm
from functools import wraps
import numpy as np


__all__ = ["AdjointMeshSeq", "annotate_qoi"]


def annotate_qoi(get_qoi):
    """
    Decorator that ensures QoIs are annotated
    properly.

    Should be applied to the :meth:`get_qoi`
    method of an :class:`AdjointMeshSeq`.
    """

    @wraps(get_qoi)
    def wrap_get_qoi(mesh_seq, solution_map, i):
        qoi = get_qoi(mesh_seq, solution_map, i)

        # Count number of arguments
        num_kwargs = 0 if qoi.__defaults__ is None else len(qoi.__defaults__)
        num_args = qoi.__code__.co_argcount - num_kwargs
        if num_args == 0:
            mesh_seq.qoi_type = "end_time"
        elif num_args == 1:
            mesh_seq.qoi_type = "time_integrated"
        else:
            raise ValueError(f"QoI should have 0 or 1 args, not {num_args}")

        @PETSc.Log.EventDecorator("pyroteus.AdjointMeshSeq.evaluate_qoi")
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
    An extension of :class:`MeshSeq` to account for
    solving adjoint problems on a sequence of meshes.

    For time-dependent quantities of interest, the
    solver should access and modify :attr:`J`, which
    holds the QoI value.
    """

    def __init__(self, time_partition, initial_meshes, get_qoi=None, **kwargs):
        """
        :kwarg get_qoi: a function, with two arguments,
            a :class:`AdjointMeshSeq`, which returns
            a function of either one or two variables,
            corresponding to either an end time or time
            integrated quantity of interest, respectively,
            as well as an index for the :class:`MeshSeq`
        """
        self.qoi_type = kwargs.pop("qoi_type")
        self.steady = kwargs.pop("steady", False)
        super().__init__(time_partition, initial_meshes, **kwargs)
        self._get_qoi = get_qoi
        self.J = 0
        self.controls = None

    @property
    @pyadjoint.no_annotations
    def initial_condition(self):
        return super().initial_condition

    @annotate_qoi
    def get_qoi(self, solution_map, i):
        if self._get_qoi is None:
            raise NotImplementedError("get_qoi needs implementing")
        return self._get_qoi(self, solution_map, i)

    @pyadjoint.no_annotations
    @PETSc.Log.EventDecorator("pyroteus.AdjointMeshSeq.get_checkpoints")
    def get_checkpoints(self, solver_kwargs={}, run_final_subinterval=False):
        """
        Solve forward on the sequence of meshes,
        extracting checkpoints corresponding to
        the starting fields on each subinterval.

        The QoI is also evaluated.

        :kwarg solver_kwargs: additional keyword
            arguments which will be passed to
            the solver
        :kwarg run_final_subinterval: toggle
            whether to solve the PDE on the
            final subinterval
        """
        self.J = 0
        N = len(self)

        # Solve forward
        checkpoints = [self.initial_condition]
        if N == 1 and not run_final_subinterval:
            return checkpoints
        for i in range(N if run_final_subinterval else N - 1):
            sols = self.solver(i, checkpoints[i], **solver_kwargs)
            assert issubclass(sols.__class__, dict), "solver should return a dict"
            fields = set(sols.keys())
            if not set(self.fields).issubset(fields):
                diff = set(self.fields).difference(fields)
                raise ValueError(f"Missing fields {diff} from solver")
            if not fields.issubset(set(self.fields)):
                diff = fields.difference(set(self.fields))
                raise ValueError(f"More solver outputs than fields ({diff} unexpected)")
            if i < N - 1:
                checkpoints.append(
                    AttrDict(
                        {
                            field: project(sols[field], fs[i + 1])
                            for field, fs in self._fs.items()
                        }
                    )
                )

        # Account for end time QoI
        if self.qoi_type == "end_time":
            qoi = self.get_qoi(sols, N - 1)
            self.J = qoi(**solver_kwargs.get("qoi_kwargs", {}))
        return checkpoints

    @PETSc.Log.EventDecorator("pyroteus.AdjointMeshSeq.solve_adjoint")
    def solve_adjoint(
        self,
        solver_kwargs={},
        adj_solver_kwargs={},
        get_adj_values=False,
        test_checkpoint_qoi=False,
    ):
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

        :return solution: an :class:`AttrDict` containing
            solution fields and their lagged versions.
        """
        num_subintervals = len(self)
        function_spaces = self.function_spaces
        P = self.time_partition

        # Solve forward to get checkpoints and evaluate QoI
        checkpoints = self.get_checkpoints(
            solver_kwargs=solver_kwargs,
            run_final_subinterval=test_checkpoint_qoi,
        )
        if self.warn and test_checkpoint_qoi and np.isclose(float(self.J), 0.0):
            self.warning("Zero QoI. Is it implemented as intended?")
        J_chk = self.J
        self.J = 0

        # Create arrays to hold exported forward and adjoint solutions
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
                                firedrake.Function(fs, name="_".join([field, label]))
                                for j in range(P.exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(function_spaces[field])
                        ]
                        for label in labels
                    }
                )
                for field in self.fields
            }
        )

        # Wrap solver to extract controls
        solver = self.solver

        @PETSc.Log.EventDecorator("pyroteus.AdjointMeshSeq.solve_adjoint.evaluate_fwd")
        @wraps(solver)
        def wrapped_solver(i, ic, **kwargs):
            init = AttrDict(
                {field: ic[field].copy(deepcopy=True) for field in self.fields}
            )
            self.controls = [pyadjoint.Control(init[field]) for field in self.fields]
            return solver(i, init, **kwargs)

        # Clear tape
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()

        # Loop over subintervals in reverse
        seeds = None
        for i in reversed(range(num_subintervals)):

            # Annotate tape on current subinterval
            sols = wrapped_solver(i, checkpoints[i], **solver_kwargs)

            # Get seed vector for reverse propagation
            if i == num_subintervals - 1:
                if self.qoi_type == "end_time":
                    qoi = self.get_qoi(sols, i)
                    self.J = qoi(**solver_kwargs.get("qoi_kwargs", {}))
                    if self.warn and np.isclose(float(self.J), 0.0):
                        self.warning("Zero QoI. Is it implemented as intended?")
            else:
                with pyadjoint.stop_annotating():
                    for field, fs in function_spaces.items():
                        sols[field].block_variable.adj_value = project(
                            seeds[field], fs[i], adjoint=True
                        )

            # Update adjoint solver kwargs
            for field in self.fields:
                for block in self.get_solve_blocks(
                    field, subinterval=i, has_adj_sol=False
                ):
                    block.adj_kwargs.update(adj_solver_kwargs)

            # Solve adjoint problem
            with PETSc.Log.Event("pyroteus.AdjointMeshSeq.solve_adjoint.evaluate_adj"):
                m = pyadjoint.enlisting.Enlist(self.controls)
                with pyadjoint.stop_annotating():
                    with tape.marked_nodes(m):
                        tape.evaluate_adj(markings=True)
            # FIXME: Using mixed Functions as Controls not correct

            # Loop over prognostic variables
            for field, fs in function_spaces.items():

                # Get solve blocks
                solve_blocks = self.get_solve_blocks(field, subinterval=i)
                num_solve_blocks = len(solve_blocks)
                assert num_solve_blocks > 0, (
                    "Looks like no solves were written to tape!"
                    " Does the solution depend on the initial condition?"
                )
                if fs[0].ufl_element() != solve_blocks[0].function_space.ufl_element():
                    raise ValueError(
                        f"Solve block list for field {field} contains mismatching"
                        f" elements ({fs[0].ufl_element()} vs. "
                        f" {solve_blocks[0].function_space.ufl_element()})"
                    )
                if "forward_old" in solutions[field]:
                    fwd_old_idx = self.get_lagged_dependency_index(
                        field, i, solve_blocks
                    )
                else:
                    fwd_old_idx = None
                if fwd_old_idx is None and "forward_old" in solutions[field]:
                    solutions[field].pop("forward_old")

                # Detect whether we have a steady problem
                steady = self.steady or (
                    num_subintervals == 1 and num_solve_blocks == 1
                )
                if steady and "adjoint_next" in sols:
                    sols.pop("adjoint_next")

                # Extract solution data
                sols = solutions[field]
                stride = P.timesteps_per_export[i]
                if len(solve_blocks[::stride]) >= P.exports_per_subinterval[i]:
                    self.warning(
                        "More solve blocks than expected"
                        f" ({len(solve_blocks[::stride])}"
                        f" > {P.exports_per_subinterval[i]-1})"
                    )
                for j, block in zip(
                    range(P.exports_per_subinterval[i] - 1), solve_blocks[::stride]
                ):

                    # Lagged forward solution and adjoint values
                    if fwd_old_idx is not None:
                        dep = block._dependencies[fwd_old_idx]
                        sols.forward_old[i][j].assign(dep.saved_output)
                        if get_adj_values:
                            sols.adj_value[i][j].assign(dep.adj_value.function)

                    # Lagged adjoint solution
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
                                f"Cannot extract solve block {j*stride+1} > {num_solve_blocks}"
                            )

                    # Forward and adjoint solution at current timestep
                    sols.forward[i][j].assign(block._outputs[0].saved_output)
                    if block.adj_sol is not None:
                        sols.adjoint[i][j].assign(block.adj_sol)

                # Check non-zero adjoint solution/value
                if self.warn and np.isclose(norm(solutions[field].adjoint[i][0]), 0.0):
                    self.warning(
                        f"Adjoint solution for field {field} on subinterval {i} is zero."
                    )
                if (
                    self.warn
                    and get_adj_values
                    and np.isclose(norm(sols.adj_value[i][0]), 0.0)
                ):
                    self.warning(
                        f"Adjoint action for field {field} on subinterval {i} is zero."
                    )

            # Get adjoint action
            seeds = {
                field: firedrake.Function(
                    function_spaces[field][i], val=control.block_variable.adj_value
                )
                for field, control in zip(self.fields, self.controls)
            }
            for field, seed in seeds.items():
                if self.warn and np.isclose(norm(seed), 0.0):
                    self.warning(
                        f"Adjoint action for field {field} on subinterval {i} is zero."
                    )
                    if steady:
                        self.warning(
                            "  You seem to have a steady-state problem. Presumably it is linear?"
                        )
            tape.clear_tape()

        # Check the QoI value agrees with that due to the checkpointing run
        if self.qoi_type == "time_integrated" and test_checkpoint_qoi:
            assert np.isclose(J_chk, self.J), (
                "QoI values computed during checkpointing and annotated"
                f" run do not match ({J_chk} vs. {self.J})"
            )
        return solutions
