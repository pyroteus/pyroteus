from firedrake import *
from pyroteus_adjoint import *
import unittest


def get_function_spaces(mesh):
    return {}


def get_form(mesh_seq):
    def form(index, sols):
        return {}

    return form


def get_bcs(mesh_seq):
    def bcs(index):
        return []

    return bcs


def get_solver(mesh_seq):
    def solver(index, ic):
        return {}

    return solver


class TestMeshSeq(unittest.TestCase):
    """
    Unit tests for :meth:`MeshSeq.fixed_point_iteration`.
    """

    def setUp(self):
        self.parameters = AdaptParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def mesh_seq(self, time_partition, mesh):
        return MeshSeq(
            time_partition,
            mesh,
            get_function_spaces=get_function_spaces,
            get_form=get_form,
            get_bcs=get_bcs,
            get_solver=get_solver,
            parameters=self.parameters,
        )

    def test_convergence_noop(self):
        def adaptor(mesh_seq, sols):
            return [False]

        miniter = self.parameters.miniter
        mesh_seq = self.mesh_seq(TimeInstant([]), UnitTriangleMesh())
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), miniter + 1)

    def test_noconvergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False]

        maxiter = self.parameters.maxiter
        mesh_seq = self.mesh_seq(TimeInstant([]), mesh2)
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), maxiter + 1)

    def test_dropout(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        mesh_seq = self.mesh_seq(time_partition, mesh2)

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)


# TODO: GoalOrientedMeshSeq version
#  - tweak the qoi function and the indicator_fn
#  - Test the new 'drop out' functionality
