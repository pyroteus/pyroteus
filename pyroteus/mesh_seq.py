"""
Sequences of meshes corresponding to a :class:`TimePartition`.
"""
import firedrake
from .utility import AttrDict, Mesh
from .interpolation import project
from collections import Iterable


__all__ = ["MeshSeq"]


class MeshSeq(object):
    """
    A sequence of meshes for solving a PDE associated
    with a particular :class:`TimePartition` of the
    temporal domain.
    """
    def __init__(self, time_partition, initial_meshes, get_function_spaces,
                 get_initial_condition, get_solver, warnings=True):
        """
        :arg time_partition: the :class:`TimePartition` which
            partitions the temporal domain
        :arg initial_meshes: list of meshes corresponding to
            the subintervals of the :class:`TimePartition`,
            or a single mesh to use for all subintervals
        :arg get_function_spaces: a function, whose only
            argument is a :class:`MeshSeq`, which constructs
            prognostic :class:`FunctionSpace` s for each
            subinterval
        :arg get_initial_condition: a function, whose only
            argument is a :class:`MeshSeq`, which specifies
            initial conditions on the first mesh
        :arg get_solver: a function, whose only argument is
            a :class:`MeshSeq`, which returns a function
            that integrates initial data over a subinterval
        :kwarg warnings: print warnings?
        """
        self.time_partition = time_partition
        self.fields = time_partition.fields
        self.subintervals = time_partition.subintervals
        self.num_subintervals = time_partition.num_subintervals
        self.meshes = initial_meshes
        if not isinstance(self.meshes, Iterable):
            self.meshes = [Mesh(initial_meshes) for subinterval in self.subintervals]
        self._fs = None
        if get_function_spaces is not None:
            self._get_function_spaces = get_function_spaces
        if get_initial_condition is not None:
            self._get_initial_condition = get_initial_condition
        if get_solver is not None:
            self._get_solver = get_solver
        self.warn = warnings

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, i):
        return self.meshes[i]

    def __setitem__(self, i, mesh):
        self.meshes[i] = mesh

    def get_function_spaces(self, mesh):
        return self._get_function_spaces(mesh)

    def get_initial_condition(self):
        return self._get_initial_condition(self)

    def get_solver(self):
        return self._get_solver(self)

    @property
    def _function_spaces_consistent(self):
        consistent = len(self.time_partition) == len(self)
        consistent &= all(len(self) == len(self._fs[field]) for field in self.fields)
        for field in self.fields:
            consistent &= all(
                mesh == fs.mesh()
                for mesh, fs in zip(self.meshes, self._fs[field])
            )
            consistent &= all(
                self._fs[field][0].ufl_element() == fs.ufl_element()
                for fs in self._fs[field]
            )
        return consistent

    @property
    def function_spaces(self):
        if self._fs is None or not self._function_spaces_consistent:
            self._fs = [self.get_function_spaces(mesh) for mesh in self.meshes]
            self._fs = AttrDict({
                field: [
                    self._fs[i][field]
                    for i in range(len(self))
                ] for field in self.fields
            })
        assert self._function_spaces_consistent, "Meshes and function spaces are inconsistent"
        return self._fs

    @property
    def initial_condition(self):
        ic = self.get_initial_condition()
        assert issubclass(ic.__class__, dict), "`get_initial_condition` should return a dict"
        assert set(self.fields).issubset(set(ic.keys())), "missing fields in initial condition"
        assert set(ic.keys()).issubset(set(self.fields)), "more initial conditions than fields"
        return AttrDict(ic)

    @property
    def solver(self):
        return self.get_solver()

    def get_checkpoints(self, solver_kwargs={}):
        """
        Solve forward on the sequence of meshes,
        extracting checkpoints corresponding to
        the starting fields on each subinterval.

        :kwarg solver_kwargs: additional keyword
            arguments which will be passed to
            the solver
        """
        solver = self.solver
        checkpoints = [self.initial_condition]
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
        return checkpoints

    def get_lagged_dependency_index(self, field, index, solve_blocks, warned=False):
        """
        Get the dependency index corresponding
        to the lagged forward solution for a
        solve block.

        :arg solve_blocks: list of taped
            :class:`GenericSolveBlocks`
        """
        fs = self.function_spaces[field][index]
        fwd_old_idx = [
            dep_index
            for dep_index, dep in enumerate(solve_blocks[0]._dependencies)
            if hasattr(dep.output, 'function_space')
            and dep.output.function_space() == solve_blocks[0].function_space == fs
        ]
        if len(fwd_old_idx) == 0:
            if not warned:
                print("WARNING: Solve block has no dependencies")  # FIXME
                warned = True
            fwd_old_idx = None
        else:
            if len(fwd_old_idx) > 1 and not warned:
                print("WARNING: Solve block has dependencies in the prognostic space"
                      + " other\n  than the PDE solution at the previous timestep."
                      + f" (Dep indices {fwd_old_idx}).\n  Naively assuming the first"
                      + " to be the right one.")  # FIXME
                warned = True
            fwd_old_idx = fwd_old_idx[0]
        return fwd_old_idx, warned

    def solve_forward(self, solver_kwargs={}):
        """
        Solve a forward problem on a sequence of subintervals.

        A dictionary of solution fields is computed, the contents
        of which give values at all exported timesteps, indexed
        first by the field label and then by type. The contents
        of these nested dictionaries are lists which are indexed
        first by subinterval and then by export. For a given
        exported timestep, the solution types are:

        * ``'forward'``: the forward solution after taking the
            timestep;
        * ``'forward_old'``: the forward solution before taking
            the timestep.

        :kwarg solver_kwargs: a dictionary providing parameters
            to the solver. Any keyword arguments for the QoI
            should be included as a subdict with label 'qoi_kwargs'

        :return solution: an :class:`AttrDict` containing
            solution fields and their lagged versions.
        """
        from firedrake_adjoint import pyadjoint
        num_subintervals = len(self)
        function_spaces = self.function_spaces

        # Create arrays to hold exported forward solutions
        solutions = AttrDict({
            field: AttrDict({
                label: [
                    [
                        firedrake.Function(fs)
                        for j in range(self.time_partition.exports_per_subinterval[i]-1)
                    ] for i, fs in enumerate(function_spaces[field])
                ] for label in ('forward', 'forward_old')
            }) for field in self.fields
        })

        # Wrap solver to extract controls
        solver = self.solver

        # Clear tape
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()

        checkpoint = self.initial_condition
        warned = not self.warn
        for i in range(num_subintervals):

            # Annotate tape on current subinterval
            checkpoint = solver(i, checkpoint, **solver_kwargs)

            # Loop over prognostic variables
            for field, fs in function_spaces.items():

                # Get solve blocks
                solve_blocks = self.time_partition.get_solve_blocks(
                    field, subinterval=i, has_adj_sol=False,
                )
                num_solve_blocks = len(solve_blocks)
                assert num_solve_blocks > 0, "Looks like no solves were written to tape!" \
                                             + " Does the solution depend on the initial condition?"
                if 'forward_old' in solutions[field]:
                    fwd_old_idx, warned = self.get_lagged_dependency_index(
                        field, i, solve_blocks, warned=warned,
                    )
                else:
                    fwd_old_idx = None
                if fwd_old_idx is None and 'forward_old' in solutions[field]:
                    solutions[field].pop('forward_old')

                # Extract solution data
                sols = solutions[field]
                stride = self.time_partition.timesteps_per_export[i]
                for j, block in enumerate(solve_blocks[::stride]):
                    sols.forward[i][j].assign(block._outputs[0].saved_output)
                    if fwd_old_idx is not None:
                        sols.forward_old[i][j].assign(block._dependencies[fwd_old_idx].saved_output)

        return solutions
