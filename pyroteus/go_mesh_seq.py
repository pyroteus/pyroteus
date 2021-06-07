from .utility import *
from .mesh_seq import MeshSeq
from pyadjoint import no_annotations
from collections import Iterable


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
        self.qoi_type = None
        self.J = 0
        self.controls = None

    @property
    @no_annotations
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

    @no_annotations
    def get_checkpoints(self, solver_kwargs={}):
        """
        Solve forward on the sequence of meshes,
        extracting checkpoints corresponding to
        the starting fields on each subinterval.

        :kwarg solver_kwargs: additional keyword
            arguments which will be passed to
            the solver
        """

        # Prepare QoI
        qoi = self.qoi
        assert self.qoi_type in ('end_time', 'time_integrated')
        if self.qoi_type = 'time_integrated':
            solver_kwargs['qoi'] = lambda *args, **kwargs: assemble(qoi(*args, **kwargs))
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
            self.J = assemble(qoi(sols))  # NOTE: any kwargs will use the default
        return checkpoints

    def solve_adjoint(self, **kwargs):
        from .adjoint import solve_adjoint
        solve_adjoint(self.solver, self.initial_condition, self.qoi, self.function_spaces, self.time_partition, **kwargs)
