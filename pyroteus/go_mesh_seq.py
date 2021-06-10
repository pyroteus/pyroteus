from .adjoint import AdjointMeshSeq

__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`MeshSeq` to account for
    goal-oriented problems.

    For goal-oriented problems, the solver should
    access and modify :attr:`J`, which holds the
    QoI value.
    """
    # TODO
