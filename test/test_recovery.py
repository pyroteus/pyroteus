"""
Test derivative recovery techniques.
"""
from firedrake import *
from pyroteus import *
from sensors import bowl, mesh_for_sensors
from parameterized import parameterized
import unittest


# ---------------------------
# standard tests for pytest
# ---------------------------


class TestRecoverySetup(unittest.TestCase):
    """
    Unit tests for derivative recovery.
    """

    def setUp(self):
        self.mesh = mesh_for_sensors(2, 4)
        self.expr = bowl(*self.mesh.coordinates)

    def get_func_ones(self, family, degree):
        V = FunctionSpace(self.mesh, family, degree)
        return Function(V).assign(1.0)

    def test_clement_space_error(self):
        f = self.get_func_ones("RT", 1)
        kwargs = dict(method="Clement", mesh=self.mesh)
        with self.assertRaises(ValueError) as cm:
            recover_hessian(f, **kwargs)
        msg = "Clement can only be used to compute gradients of Lagrange fields."
        self.assertEqual(str(cm.exception), msg)

    def test_clement_degree_error(self):
        f = self.get_func_ones("DG", 0)
        kwargs = dict(method="Clement", mesh=self.mesh)
        with self.assertRaises(ValueError) as cm:
            recover_hessian(f, **kwargs)
        msg = "Clement can only be used to compute gradients of fields of degree > 0."
        self.assertEqual(str(cm.exception), msg)

    def test_zz_notimplemented_error(self):
        f = self.get_func_ones("CG", 1)
        kwargs = dict(method="ZZ", mesh=self.mesh)
        with self.assertRaises(NotImplementedError) as cm:
            recover_hessian(f, **kwargs)
        msg = "Zienkiewicz-Zhu recovery not yet implemented."
        self.assertEqual(str(cm.exception), msg)

    def test_unrecognised_interior_method_error(self):
        f = self.get_func_ones("CG", 1)
        kwargs = dict(method="some_method", mesh=self.mesh)
        with self.assertRaises(ValueError) as cm:
            recover_hessian(f, **kwargs)
        msg = "Recovery method 'some_method' not recognised."
        self.assertEqual(str(cm.exception), msg)

    def test_unrecognised_boundary_method_error(self):
        f = self.get_func_ones("CG", 1)
        d = {"interior": f, 1: f}
        kwargs = dict(method="some_method")
        with self.assertRaises(ValueError) as cm:
            recover_boundary_hessian(d, self.mesh, **kwargs)
        msg = (
            "Recovery method 'some_method' not supported for Hessians on the boundary."
        )
        self.assertEqual(str(cm.exception), msg)


class TestRecoveryBowl(unittest.TestCase):
    """
    Unit tests for recovery applied to a quadratic 'bowl' sensor.
    """

    @staticmethod
    def relative_error(approx, ignore_boundary=False, norm_type="L2"):
        mesh = approx.function_space().mesh()
        dim = mesh.topological_dimension()
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        I = interpolate(Identity(dim), P1_ten)

        # Check that they agree
        cond = Constant(1.0)
        if ignore_boundary:
            x = SpatialCoordinate(mesh)
            cond = And(x[0] > -0.8, x[0] < 0.8)
            for i in range(1, dim):
                cond = And(cond, And(x[i] > -0.8, x[i] < 0.8))
            cond = conditional(cond, 1, 0)
        err = errornorm(approx, I, norm_type=norm_type, condition=cond)
        err /= norm(I, norm_type=norm_type, condition=cond)
        return err

    @parameterized.expand(
        [
            (2, "l2"),
            (2, "L1"),
            (2, "L2"),
            (3, "l2"),
            (3, "L1"),
            (3, "L2"),
        ]
    )
    def test_interior_L2_mixed(self, dim, norm_type):
        # TODO: parallel version
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        H = recover_hessian(f, method="L2", mesh=mesh, mixed=True)
        err = self.relative_error(H, norm_type=norm_type)
        assert err < 1.0e-05

    @parameterized.expand(
        [
            (2, "l2"),
            (2, "L1"),
            (2, "L2"),
            (3, "l2"),
            (3, "L1"),
            (3, "L2"),
        ]
    )
    def test_interior_L2_target_spaces(self, dim, norm_type):
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        V = VectorFunctionSpace(mesh, "CG", 1)
        W = TensorFunctionSpace(mesh, "CG", 1)
        kwargs = dict(mesh=mesh, target_spaces=(V, W))
        H = recover_hessian(f, method="L2", **kwargs)
        err = self.relative_error(H, norm_type=norm_type)
        assert err < 1.0e-05

    @parameterized.expand(
        [
            (2, "l2"),
            (2, "L1"),
            (2, "L2"),
            (3, "l2"),
            (3, "L1"),
            (3, "L2"),
        ]
    )
    def test_interior_Clement(self, dim, norm_type):
        mesh = mesh_for_sensors(dim, 20)
        f = interpolate(bowl(*mesh.coordinates), FunctionSpace(mesh, "CG", 1))
        H = recover_hessian(f, method="Clement", mesh=mesh, mixed=True)
        err = self.relative_error(H, norm_type=norm_type, ignore_boundary=True)
        assert err < 1.0e-05

    # TODO: test_interior_Clement_quadratic_exact

    @parameterized.expand([(2,)])
    def test_boundary_L2(self, dim):
        # FIXME: 3D case for test_boundary_L2
        mesh = mesh_for_sensors(dim, 4)
        f = bowl(*mesh.coordinates)
        tags = list(mesh.exterior_facets.unique_markers) + ["interior"]
        f = {i: f for i in tags}
        H = recover_boundary_hessian(f, mesh, method="L2")

        # Check its directional derivatives in boundaries are zero
        S = construct_orthonormal_basis(FacetNormal(mesh))
        for s in S:
            dHds = abs(assemble(dot(div(H), s) * ds))
            assert dHds < 2.0e-08, "Non-zero tangential derivative"

    @parameterized.expand([(2,)])
    def test_boundary_Clement(self, dim):
        # FIXME: 3D case for test_boundary_Clement
        mesh = mesh_for_sensors(dim, 20)
        f = bowl(*mesh.coordinates)
        tags = list(mesh.exterior_facets.unique_markers) + ["interior"]
        f = {i: f for i in tags}
        H = recover_boundary_hessian(f, mesh, method="Clement")

        # Check its directional derivatives in boundaries are zero
        S = construct_orthonormal_basis(FacetNormal(mesh))
        for s in S:
            dHds = abs(assemble(dot(div(H), s) * ds))
            assert dHds < 2.0e-08, "Non-zero tangential derivative"


# TODO: Implement a difficult recovery test that force use of an LU solver.
