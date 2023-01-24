from firedrake import *
from pyroteus import *
import pyroteus.quality as mq
from parameterized import parameterized
import pytest
import unittest
from utility import uniform_mesh


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_facet_areas(dim):
    """
    Check that the computation of facet areas
    sums to the expected value for a uniform
    (isotropic) triangular or tetrahedral mesh.
    """
    mesh = uniform_mesh(dim, 1)
    f = mq.get_facet_areas2d if dim == 2 else mq.get_facet_areas3d
    expected = 5.414214 if dim == 2 else 10.242641
    assert np.isclose(sum(f(mesh).dat.data), expected)


class TestQuality(unittest.TestCase):
    """
    Unit tests for quality metrics.
    """

    def quality(self, name, mesh, **kwargs):
        dim = mesh.topological_dimension()
        measure = getattr(mq, name)
        if name.startswith("get_quality_metrics"):
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)
            M = interpolate(Identity(dim), P1_ten)
            return measure(mesh, M, **kwargs)
        return measure(mesh, **kwargs)

    @parameterized.expand(
        [
            ("get_min_angles2d", np.pi / 4),
            ("get_areas2d", 0.005),
            ("get_eskews2d", 1.070796),
            ("get_aspect_ratios2d", 1.207107),
            ("get_scaled_jacobians2d", 0.707107),
            ("get_skewnesses2d", 0.463648),
            ("get_quality_metrics2d", 6.928203),
        ]
    )
    def test_uniform_quality_2d(self, measure, expected):
        mesh = uniform_mesh(2, 10)
        q = self.quality(measure, mesh)
        truth = Function(q.function_space()).assign(expected)
        self.assertAlmostEqual(errornorm(truth, q), 0.0, places=6)
        if measure == "get_areas2d":
            s = q.vector().gather().sum()
            self.assertAlmostEqual(s, 1.0)

    @parameterized.expand(
        [
            ("get_min_angles3d", 0.61547971),
            ("get_volumes3d", 0.00260417),
            ("get_eskews3d", 0.41226017),
            ("get_aspect_ratios3d", 1.39384685),
            ("get_scaled_jacobians3d", 0.40824829),
            ("get_quality_metrics3d", 1.25),
        ]
    )
    def test_uniform_quality_3d(self, measure, expected):
        mesh = uniform_mesh(3, 4)
        q = self.quality(measure, mesh)
        truth = Function(q.function_space()).assign(expected)
        self.assertAlmostEqual(errornorm(truth, q), 0.0)
        if measure == "get_volumes3d":
            s = q.vector().gather().sum()
            self.assertAlmostEqual(s, 1.0)

    @parameterized.expand(
        [
            ("get_areas2d"),
            ("get_aspect_ratios2d"),
            ("get_scaled_jacobians2d"),
        ]
    )
    def test_consistency_2d(self, measure):
        np.random.seed(0)
        mesh = uniform_mesh(2, 4)
        mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
        quality_cpp = self.quality(measure, mesh, python=False)
        quality_py = self.quality(measure, mesh, python=True)
        self.assertAlmostEqual(errornorm(quality_cpp, quality_py), 0.0)

    @parameterized.expand([("get_volumes3d")])
    def test_consistency_3d(self, measure):
        np.random.seed(0)
        mesh = uniform_mesh(3, 4)
        mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
        quality_cpp = self.quality(measure, mesh, python=False)
        quality_py = self.quality(measure, mesh, python=True)
        self.assertAlmostEqual(errornorm(quality_cpp, quality_py), 0.0)

    @parameterized.expand(
        [
            ("get_facet_areas2d"),
            ("get_facet_areas3d"),
            ("get_skewnesses3d"),
        ]
    )
    def test_cxx_notimplemented(self, measure):
        mesh = uniform_mesh(2, 10)
        with pytest.raises(NotImplementedError):
            self.quality(measure, mesh, python=False)

    @parameterized.expand(
        [
            ("get_min_angles2d"),
            ("get_min_angles3d"),
            ("get_aspect_ratios3d"),
            ("get_eskews2d"),
            ("get_eskews3d"),
            ("get_skewnesses2d"),
            ("get_skewnesses3d"),
            ("get_scaled_jacobians3d"),
            ("get_quality_metrics2d"),
            ("get_quality_metrics3d"),
        ]
    )
    def test_python_notimplemented(self, measure):
        mesh = uniform_mesh(2, 10)
        with pytest.raises(NotImplementedError):
            self.quality(measure, mesh, python=True)
