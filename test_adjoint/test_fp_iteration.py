from firedrake import *
from goalie_adjoint import *
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


def get_qoi(mesh_seq, solutions, index):
    def qoi():
        if mesh_seq.fp_iteration % 2 == 0:
            return Constant(1, domain=mesh_seq[index]) * dx
        else:
            return Constant(2, domain=mesh_seq[index]) * dx

    return qoi


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
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

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
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_dropout(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        mesh_seq = self.mesh_seq(time_partition, mesh2)

        def adaptor(mesh_seq, sols):
            mesh_seq[1] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [True, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [False, True]))

    def test_no_late_convergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        mesh_seq = self.mesh_seq(time_partition, mesh2)

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [False, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [True, True]))


class TestGoalOrientedMeshSeq(unittest.TestCase):
    """
    Unit tests for :meth:`GoalOrientedMeshSeq.fixed_point_iteration`.
    """

    def setUp(self):
        self.parameters = GoalOrientedParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def mesh_seq(self, time_partition, mesh, qoi_type="steady"):
        return GoalOrientedMeshSeq(
            time_partition,
            mesh,
            get_function_spaces=get_function_spaces,
            get_form=get_form,
            get_bcs=get_bcs,
            get_solver=get_solver,
            get_qoi=get_qoi,
            parameters=self.parameters,
            qoi_type=qoi_type,
        )

    def test_convergence_noop(self):
        def adaptor(mesh_seq, sols, indicators):
            return [False]

        miniter = self.parameters.miniter
        mesh_seq = self.mesh_seq(TimeInstant([]), UnitTriangleMesh())
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), miniter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_noconvergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False]

        maxiter = self.parameters.maxiter
        mesh_seq = self.mesh_seq(TimeInstant([]), mesh2)
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), maxiter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_dropout(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        mesh_seq = self.mesh_seq(time_partition, mesh2, qoi_type="end_time")

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[1] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [True, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [False, True]))

    def test_no_late_convergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        mesh_seq = self.mesh_seq(time_partition, mesh2, qoi_type="end_time")

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [False, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [True, True]))
