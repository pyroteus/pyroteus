from .utility import *
from pyadjoint import no_annotations
from collections import Iterable


__all__ = ["MeshSeq", "GoalOrientedMeshSeq"]


class MeshSeq(object):
    """
    A sequence of meshes for solving a PDE associated
    with a particular :class:`TimePartition` of the
    temporal domain.
    """
    def __init__(self, time_partition, initial_meshes, get_function_spaces,
                 get_initial_condition, get_solver):
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
            a :class:`MeshSeq`, which integrates initial
            data over a subinterval
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

    def __len__(self):
        return len(self.time_partition)

    def __getitem__(self, i):
        return self.meshes[i]

    def __setitem__(self, i, mesh):
        self.meshes[i] = mesh

    @property
    def function_spaces(self):
        if self._fs is None or not all(mesh == fs.mesh() for mesh, fs in zip(self.meshes, self._fs)):
            self._fs = self.get_function_spaces()
        return self._fs

    @property
    @no_annotations
    def initial_condition(self):
        return self.get_initial_condition()

    @property
    def solver(self):
        return self.get_solver()


class GoalOrientedMeshSeq(MeshSeq):
    """
    An extension of :class:`MeshSeq` to account for
    goal-oriented problems.
    """
    def __init__(self, time_partition, initial_meshes, get_function_spaces,
                 get_initial_condition, get_solver, get_qoi):
        """
        :arg get_qoi: a function, whose only argument is
            a :class:`GoalOrientedMeshSeq`, which returns
            a function of either one or two variables,
            corresponding to either an end time or time
            integrated quantity of interest, respectively
        """
        super(GoalOrientedMeshSeq, self).__init__(
            time_partition, initial_meshes, get_function_spaces, get_initial_condition, get_solver,
        )
        self.get_qoi = get_qoi

    @property
    def qoi(self):
        return self.get_qoi()

    def solve_adjoint(self, **kwargs):  # TODO: base solver on self
        from .adjoint import solve_adjoint
        solve_adjoint(self.solver, self.initial_condition, self.qoi, self.function_spaces, self.time_partition, **kwargs)
