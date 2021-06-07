import firedrake
from firedrake_adjoint import Control
import pyadjoint
from .interpolation import project
from .mesh_seq import MeshSeq
from .utility import AttrDict, norm
from functools import wraps
import numpy as np


__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(MeshSeq):
    """
    An extension of :class:`MeshSeq` to account for
    goal-oriented problems.

    For goal-oriented problems, the solver should
    access and modify :attr:`J`, which holds the
    QoI value.
    """
    def __init__(self, time_partition, initial_meshes, get_function_spaces,
                 get_initial_condition, get_solver, get_qoi, **kwargs):
        """
        :arg get_qoi: a function, whose only argument is
            a :class:`GoalOrientedMeshSeq`, which returns
            a function of either one or two variables,
            corresponding to either an end time or time
            integrated quantity of interest, respectively
        """
        super(GoalOrientedMeshSeq, self).__init__(
            time_partition, initial_meshes, get_function_spaces,
            get_initial_condition, get_solver, **kwargs
        )
        self.get_qoi = get_qoi
        self.qoi_type = None
        self.J = 0
        self.controls = None
        self.steady = kwargs.get('steady', False)

    @property
    @pyadjoint.no_annotations
    def initial_condition(self):
        return super(GoalOrientedMeshSeq, self).initial_condition

    @property
    def qoi(self):
        qoi = self.get_qoi()
        num_kwargs = 0 if qoi.__defaults__ is None else len(qoi.__defaults__)
        if num_kwargs > 0:
            print("WARNING: QoI has kwargs which will be unused")
        num_args = qoi.__code__.co_argcount - num_kwargs
        if num_args == 1:
            self.qoi_type = 'end_time'
        elif num_args == 2:
            self.qoi_type = 'time_integrated'
        else:
            raise ValueError(f"QoI should have 1 or 2 args, not {num_args}")
        return qoi

    @pyadjoint.no_annotations
    def get_checkpoints(self, solver_kwargs={}):
        """
        Solve forward on the sequence of meshes,
        extracting checkpoints corresponding to
        the starting fields on each subinterval.

        The QoI is also evaluated.

        :kwarg solver_kwargs: additional keyword
            arguments which will be passed to
            the solver
        """

        # Prepare QoI
        qoi = self.qoi
        assert self.qoi_type in ('end_time', 'time_integrated')
        if self.qoi_type == 'time_integrated':
            solver_kwargs['qoi'] = lambda *args, **kwargs: firedrake.assemble(qoi(*args, **kwargs))
        self.J = 0

        # Solve forward
        checkpoints = [self.initial_condition]
        solver = self.solver
        for i in range(len(self)):
            sols = solver(self, checkpoints[i], *self.time_partition[i], **solver_kwargs)
            assert issubclass(sols.__class__, dict), "solver should return a dict"
            assert set(self.fields).issubclass(set(sols.keys())), "missing fields from solver"
            assert set(sols.keys()).issubclass(set(self.fields)), "more solver outputs than fields"
            if i < len(self)-1:
                checkpoints.append({
                    field: project(sols[field], fs[i+1])
                    for field, fs in self._fs.items()
                })

        # Account for end time QoI
        if self.qoi_type == 'end_time':
            self.J = firedrake.assemble(qoi(sols))  # NOTE: any kwargs will use the default
        return checkpoints

    @property
    def wrapped_solver(self):
        """
        Wrapper for a solver which records a
        :class:`Control` for each component of
        its initial condition.
        """
        solver = self.solver

        @wraps(solver)
        def wrapper(self, ic, t_start, t_end, dt, **kwargs):
            init = {field: ic[field].copy(deepcopy=True) for field in self.fields}
            self.controls = [Control(init[field]) for field in self.fields]
            return solver(self, init, t_start, t_end, dt, **kwargs)

        return wrapper

    @property
    def wrapped_qoi(self):
        """
        Wrapper for contributions to a QoI which
        sets the corresponding ``adj_value`` to
        unity.
        """
        qoi = self.qoi

        @wraps(qoi)
        def wrapper(*args, **kwargs):
            j = firedrake.assemble(qoi(*args, **kwargs))
            j.block_variable.adj_value = 1.0
            return j

        return wrapper

    def solve_adjoint(self, solver_kwargs={}, get_adj_values=False):
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
            the timestep;
        * ``'adjoint'``: the adjoint solution after taking the
            timestep;
        * ``'adjoint_next'``: the adjoint solution before taking
            the timestep (backwards).

        :kwarg solver_kwargs: a dictionary providing parameters
            to the solver
        :kwarg get_adj_values: additionally output adjoint
            actions at exported timesteps

        :return solution: an :class:`AttrDict` containing
            solution fields and their lagged versions.
        """
        num_subintervals = len(self)
        function_spaces = self.function_spaces

        # Solve forward to get checkpoints and evaluate QoI
        checkpoints = self.get_checkpoints(solver_kwargs=solver_kwargs)
        if self.warn and np.isclose(float(self.J), 0.0):
            print("WARNING: Zero QoI. Is it implemented as intended?")

        # Create arrays to hold exported forward and adjoint solutions
        labels = ('forward', 'forward_old', 'adjoint')
        if not self.steady:
            labels += ('adjoint_next',)
        solutions = AttrDict({
            field: AttrDict({
                label: [
                    [
                        firedrake.Function(fs)
                        for j in range(self.time_partition.exports_per_subinterval[i]-1)
                    ] for i, fs in enumerate(function_spaces[field])
                ] for label in labels
            }) for field in self.fields
        })

        # Create arrays to hold adjoint actions
        if get_adj_values:
            adj_values = AttrDict({
                field: [
                    [
                        firedrake.Function(fs)
                        for j in range(self.time_partition.exports_per_subinterval[i]-1)
                    ] for i, fs in enumerate(function_spaces[field])
                ] for field in self.fields
            })
        else:
            adj_values = None

        # Wrap solver and QoI
        wrapped_solver = self.wrapped_solver
        wrapped_qoi = self.wrapped_qoi
        if self.qoi_type == 'time_integrated':
            solver_kwargs['qoi'] = wrapped_qoi
        assert hasattr(self, 'controls')

        # Clear tape
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()

        # Loop over subintervals in reverse
        seeds = None
        warned = not self.warn
        for i in reversed(range(num_subintervals)):

            # Annotate tape on current subinterval
            sols = wrapped_solver(checkpoints[i], *self.time_partition[i], **solver_kwargs)

            # Get seed vector for reverse propagation
            if i == num_subintervals-1:
                if self.qoi_type == 'end_time':
                    self.J = wrapped_qoi(sols)  # NOTE: any kwargs will use the default
                    if self.warn and np.isclose(float(self.J), 0.0):
                        print("WARNING: Zero QoI. Is it implemented as intended?")
            else:
                with pyadjoint.stop_annotating():
                    for field, fs in function_spaces.items():
                        sols[field].block_variable.adj_value = project(seeds[field], fs[i], adjoint=True)

            # Solve adjoint problem
            m = pyadjoint.enlisting.Enlist(self.controls)
            with pyadjoint.stop_annotating():
                with tape.marked_nodes(m):
                    tape.evaluate_adj(markings=True)

            # Loop over prognostic variables
            for field, fs in function_spaces.items():

                # Get solve blocks
                solve_blocks = self.time_partition.get_solve_blocks(field, subinterval=i)
                num_solve_blocks = len(solve_blocks)
                assert num_solve_blocks > 0, "Looks like no solves were written to tape!" \
                                             + " Does the solution depend on the initial condition?"

                # Detect whether we have a steady problem
                steady = self.steady or (num_subintervals == 1 and num_solve_blocks == 1)
                if steady and 'adjoint_next' in sols:
                    sols.pop('adjoint_next')

                # Get lagged forward solution dependecy index
                fwd_old_idx = [
                    dep_index
                    for dep_index, dep in enumerate(solve_blocks[0]._dependencies)
                    if hasattr(dep.output, 'function_space')
                    and dep.output.function_space() == solve_blocks[0].function_space == fs[i]
                ]
                if not warned and len(fwd_old_idx) != 1:
                    print("WARNING: Solve block has dependencies in the prognostic space other\n"
                          + "  than the PDE solution at the previous timestep. (Dep indices"
                          + f" {fwd_old_idx}).\n  Naively assuming the first to be the right one.")
                    warned = True  # FIXME
                fwd_old_idx = fwd_old_idx[0]

                # Extract solution data
                sols = solutions[field]
                stride = self.time_partition.timesteps_per_export[i]
                for j, block in enumerate(solve_blocks[::stride]):
                    sols.forward[i][j].assign(block._outputs[0].saved_output)
                    sols.adjoint[i][j].assign(block.adj_sol)
                    if adj_values is not None:
                        adj_values[field][i][j].assign(block._dependencies[fwd_old_idx].adj_value.function)
                    sols.forward_old[i][j].assign(block._dependencies[fwd_old_idx].saved_output)
                    if not steady:
                        if j*stride+1 < num_solve_blocks:
                            sols.adjoint_next[i][j].assign(solve_blocks[j*stride+1].adj_sol)
                        elif j*stride+1 == num_solve_blocks:
                            if i+1 < num_subintervals:
                                sols.adjoint_next[i][j].assign(
                                    project(sols.adjoint_next[i+1][0], fs[i], adjoint=True)
                                )
                        else:
                            raise IndexError(f"Cannot extract solve block {j*stride+1} "
                                             + f"> {num_solve_blocks}")
                if self.warn and np.isclose(norm(solutions[field].adjoint[i][0]), 0.0):
                    print(f"WARNING: Adjoint solution for field {field} on subinterval {i} is zero.")
                if self.warn and adj_values is not None and np.isclose(norm(adj_values[field][i][0]), 0.0):
                    print(f"WARNING: Adjoint action for field {field} on subinterval {i} is zero.")

            # Get adjoint action
            seeds = {
                field: firedrake.Function(function_spaces[field][i],
                                          val=control.block_variable.adj_value)
                for field, control in zip(self.fields, self.controls)
            }
            for field, seed in seeds.items():
                if self.warn and np.isclose(norm(seed), 0.0):
                    print(f"WARNING: Adjoint action for field {field} on subinterval {i} is zero.")
                    if steady:
                        print("  You seem to have a steady-state problem. Presumably it is linear?")
            tape.clear_tape()

        # Process return values
        ret = (solutions,)
        if adj_values is not None:
            ret += (adj_values,)
        return ret
