"""
Test interpolation schemes.
"""
from firedrake import *
from goalie import *
from goalie.utility import function2cofunction
from parameterized import parameterized
import unittest


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
        with self.assertRaises(NotImplementedError) as cm:
            project(2 * Function(Vs), Vt)
        msg = "Can only currently project Functions and Cofunctions."
        self.assertEqual(str(cm.exception), msg)

    def test_no_sub_source_space(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vt = Vt * Vt
        with self.assertRaises(ValueError) as cm:
            project(Function(Vs), Function(Vt))
        msg = "Target space has multiple components but source space does not."
        self.assertEqual(str(cm.exception), msg)

    def test_no_sub_source_space_adjoint(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vt = Vt * Vt
        with self.assertRaises(ValueError) as cm:
            project(Cofunction(Vs.dual()), Cofunction(Vt.dual()))
        msg = "Source space has multiple components but target space does not."
        self.assertEqual(str(cm.exception), msg)

    def test_no_sub_target_space(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vs = Vs * Vs
        with self.assertRaises(ValueError) as cm:
            project(Function(Vs), Function(Vt))
        msg = "Source space has multiple components but target space does not."
        self.assertEqual(str(cm.exception), msg)

    def test_no_sub_target_space_adjoint(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.target_mesh, "CG", 1)
        Vs = Vs * Vs
        with self.assertRaises(ValueError) as cm:
            project(Cofunction(Vs.dual()), Cofunction(Vt.dual()))
        msg = "Target space has multiple components but source space does not."
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

    def test_project_same_space(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = interpolate(self.sinusoid(), Vs)
        target = Function(Vs)
        project(source, target)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_space_adjoint(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        source = interpolate(self.sinusoid(), Vs)
        source = function2cofunction(source)
        target = Cofunction(Vs.dual())
        project(source, target)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_space_mixed(self):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        Vs = P1 * P1
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vs)
        project(source, target)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_space_mixed_adjoint(self):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        Vs = P1 * P1
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        source = function2cofunction(source)
        target = Cofunction(Vs.dual())
        project(source, target)
        expected = source
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_mesh(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = interpolate(self.sinusoid(), Vs)
        target = Function(Vt)
        project(source, target)
        expected = Function(Vt).project(source)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_mesh_adjoint(self):
        Vs = FunctionSpace(self.source_mesh, "CG", 1)
        Vt = FunctionSpace(self.source_mesh, "DG", 0)
        source = interpolate(self.sinusoid(), Vs)
        target = Cofunction(Vt.dual())
        project(function2cofunction(source), target)
        expected = function2cofunction(Function(Vt).project(source))
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_mesh_mixed(self):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        Vs = P1 * P1
        Vt = P0 * P0
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Function(Vt)
        project(source, target)
        expected = Function(Vt)
        e1, e2 = expected.subfunctions
        e1.project(s1)
        e2.project(s2)
        self.assertAlmostEqual(errornorm(expected, target), 0)

    def test_project_same_mesh_mixed_adjoint(self):
        P1 = FunctionSpace(self.source_mesh, "CG", 1)
        P0 = FunctionSpace(self.source_mesh, "DG", 0)
        Vs = P1 * P1
        Vt = P0 * P0
        source = Function(Vs)
        s1, s2 = source.subfunctions
        s1.interpolate(self.sinusoid())
        s2.interpolate(-self.sinusoid())
        target = Cofunction(Vt.dual())
        project(function2cofunction(source), target)
        expected = Function(Vt)
        e1, e2 = expected.subfunctions
        e1.project(s1)
        e2.project(s2)
        self.assertAlmostEqual(errornorm(expected, target), 0)
