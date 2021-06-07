from .utility import Mesh
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
            prognostic :class:`FunctionSpace`s for each
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
        self.meshes = initial_meshes
        if not isinstance(self.meshes, Iterable):
            self.meshes = [Mesh(initial_meshes) for subinterval in self.subintervals]
        self._fs = None
        self.get_function_spaces = get_function_spaces
        self.get_initial_condition = get_initial_condition
        self.get_solver = get_solver
        self.warn = warnings

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, i):
        return self.meshes[i]

    def __setitem__(self, i, mesh):
        self.meshes[i] = mesh

    @property
    def _function_spaces_consistent(self):
        consistent = len(self.time_partition) == len(self) == len(self._fs)
        for field in self.fields:
            consistent &= all(
                mesh == fs.mesh()
                for mesh, fs in zip(self.meshes, self._fs[field])
            )
        return consistent

    @property
    def function_spaces(self):
        if self._fs is None or not self._function_spaces_consistent:
            self._fs = self.get_function_spaces()
        assert self._function_spaces_consistent, "Meshes and function spaces are inconsistent"
        return self._fs

    @property
    def initial_condition(self):
        ic = self.get_initial_condition()
        assert issubclass(ic.__class__, dict), "`get_initial_condition` should return a dict"
        assert set(self.fields).issubclass(set(ic.keys())), "missing fields in initial condition"
        assert set(ic.keys()).issubclass(set(self.fields)), "more initial conditions than fields"
        return ic

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
