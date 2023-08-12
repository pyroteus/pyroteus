"""
Test interpolation schemes.
"""
from firedrake import *
from goalie import *
from parameterized import parameterized
import unittest


class TestClement(unittest.TestCase):
    """
    Unit tests for Clement interpolant.
    """

    def setUp(self):
        n = 5
        self.mesh = UnitSquareMesh(n, n)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)

        h = 1 / n
        self.x, self.y = SpatialCoordinate(self.mesh)
        self.interior = conditional(
            And(And(self.x > h, self.x < 1 - h), And(self.y > h, self.y < 1 - h)), 1, 0
        )
        self.boundary = 1 - self.interior
        self.corner = conditional(
            And(Or(self.x < h, self.x > 1 - h), Or(self.y < h, self.y > 1 - h)), 1, 0
        )

    def get_space(self, rank, family, degree):
        if rank == 0:
            return FunctionSpace(self.mesh, family, degree)
        elif rank == 1:
            return VectorFunctionSpace(self.mesh, family, degree)
        else:
            shape = tuple(rank * [self.mesh.topological_dimension()])
            return TensorFunctionSpace(self.mesh, family, degree, shape=shape)

    def analytic(self, rank):
        if rank == 0:
            return self.x
        elif rank == 1:
            return as_vector((self.x, self.y))
        else:
            return as_matrix([[self.x, self.y], [-self.y, -self.x]])

    def test_rank_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.get_space(3, "DG", 0)))
        msg = "Rank-4 tensors are not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_source_space_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.get_space(0, "CG", 1)))
        msg = "Source function provided must be from a P0 space."
        self.assertEqual(str(cm.exception), msg)

    def test_target_space_error(self):
        with self.assertRaises(ValueError) as cm:
            clement_interpolant(Function(self.P0), target_space=self.P0)
        msg = "Target space provided must be P1."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[0], [1], [2]])
    def test_volume_average_2d(self, rank):
        exact = self.analytic(rank)
        P0 = self.get_space(rank, "DG", 0)
        P1 = self.get_space(rank, "CG", 1)
        source = Function(P0).project(exact)
        target = clement_interpolant(source)
        expected = Function(P1).interpolate(exact)
        err = assemble(self.interior * (target - expected) ** 2 * dx)
        self.assertAlmostEqual(err, 0)

    @parameterized.expand([[0], [1], [2]])
    def test_facet_average_2d(self, rank):
        exact = self.analytic(rank)
        P0 = self.get_space(rank, "DG", 0)
        source = Function(P0).project(exact)
        target = clement_interpolant(source, boundary=True)
        expected = source
        integrand = (1 - self.corner) * (target - expected) ** 2

        # Check approximate recovery
        for tag in [1, 2, 3, 4]:
            self.assertLess(assemble(integrand * ds(tag)), 5e-3)


class TestProject(unittest.TestCase):
    """
    Unit tests for mesh-to-mesh projection.
    """

    def setUp(self):
        self.source_mesh = UnitSquareMesh(1, 1, diagonal="left")
        self.target_mesh = UnitSquareMesh(1, 1, diagonal="right")

    def sinusoid(self, source=True):
        x, y = SpatialCoordinate(self.source_mesh if source else self.target_mesh)
        return sin(pi * x) * sin(pi * y)

    def test_notimplemented_error(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        source = Function(Vs)
        with self.assertRaises(NotImplementedError) as cm:
            project(2 * source, Vt)
        msg = "Can only currently project Functions."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[False], [True]])
    def test_no_sub_source_space(self, adjoint):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vt = Vt * Vt
        with self.assertRaises(ValueError) as cm:
            project(Function(Vs), Function(Vt), adjoint=adjoint)
        if adjoint:
            msg = "Source space has multiple components but target space does not."
        else:
            msg = "Target space has multiple components but source space does not."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[False], [True]])
    def test_no_sub_target_space(self, adjoint):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vs = Vs * Vs
        with self.assertRaises(ValueError) as cm:
            project(Function(Vs), Function(Vt), adjoint=adjoint)
        if adjoint:
            msg = "Target space has multiple components but source space does not."
        else:
            msg = "Source space has multiple components but target space does not."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_number_sub_spaces(self):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.target_mesh, "DG", 0)
        Vs = P1 * P1 * P1
        Vt = P0 * P0
        with self.assertRaises(ValueError) as cm:
            project(Function(Vs), Function(Vt))
        msg = "Inconsistent numbers of components in source and target spaces: 3 vs. 2."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[False], [True]])
    def test_project_same_space(self, adjoint):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = interpolate(self.sinusoid(), Vs)
        target = Function(Vs)
        project(source, target, adjoint=adjoint)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand([[False], [True]])
    def test_project_same_space_mixed(self, adjoint):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        Vs = P1 * P1
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vs)
        project(source, target, adjoint=adjoint)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand([[False], [True]])
    def test_project_same_mesh(self, adjoint):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = interpolate(self.sinusoid(), Vs)
        target = Function(Vt)
        project(source, target, adjoint=adjoint)
        expected = Function(Vt)
        expected.project(source)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    @parameterized.expand([[False], [True]])
    def test_project_same_mesh_mixed(self, adjoint):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        Vs = P1 * P1
        Vt = P0 * P0
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vt)
        project(source, target, adjoint=adjoint)
        expected = Function(Vt)
        e1, e2 = expected.subfunctions
        e1.project(s1)
        e2.project(s2)
        self.assertAlmostEqual(errornorm(expected, target), 0)
