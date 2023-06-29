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
        self.parameters = AdaptParameters()
        time_partition = TimeInstant([])
        self.mesh_seq = MeshSeq(
            time_partition,
            UnitTriangleMesh(),
            get_function_spaces=get_function_spaces,
            get_form=get_form,
            get_bcs=get_bcs,
            get_solver=get_solver,
            parameters=self.parameters,
        )

    def test_convergence_noop(self):
        def adaptor(mesh_seq, sols):
            pass

        miniter = self.parameters.miniter
        self.mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(self.mesh_seq.element_counts), miniter + 1)

    def test_noconvergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2

        maxiter = self.parameters.maxiter
        self.mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(self.mesh_seq.element_counts), maxiter + 1)


# TODO: GoalOrientedMeshSeq version
#  - tweak the qoi function and the indicator_fn
#  - Test the new 'drop out' functionality
