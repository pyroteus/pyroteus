"""
Test matrix decomposition par_loops.
"""
from firedrake import *
from firedrake.meshadapt import RiemannianMetric
from pyroteus import *
from utility import uniform_mesh
from parameterized import parameterized
import pytest
import unittest


class TestMetricDecompositions(unittest.TestCase):
    """
    Unit tests for metric decompositions.
    """

    def mesh(self, dim):
        return uniform_mesh(dim, 1)

    def test_compute_eigendecomposition_type_error(self):
        P1_ten = TensorFunctionSpace(self.mesh(2), "CG", 1)
        with self.assertRaises(ValueError) as cm:
            compute_eigendecomposition(Function(P1_ten))
        msg = (
            "Can only compute eigendecompositions of RiemannianMetrics,"
            " not objects of type '<class 'firedrake.function.Function'>'."
        )
        assert str(cm.exception) == msg

    def test_assemble_eigendecomposition_evectors_rank2_error(self):
        P1_vec = VectorFunctionSpace(self.mesh(2), "CG", 1)
        evalues = Function(P1_vec)
        evectors = Function(P1_vec)
        with self.assertRaises(ValueError) as cm:
            assemble_eigendecomposition(evectors, evalues)
        msg = "Eigenvector Function should be rank-2, not rank-1."
        assert str(cm.exception) == msg

    def test_assemble_eigendecomposition_evalues_rank1_error(self):
        P1_ten = TensorFunctionSpace(self.mesh(2), "CG", 1)
        evalues = Function(P1_ten)
        evectors = Function(P1_ten)
        with self.assertRaises(ValueError) as cm:
            assemble_eigendecomposition(evectors, evalues)
        msg = "Eigenvalue Function should be rank-1, not rank-2."
        assert str(cm.exception) == msg

    def test_assemble_eigendecomposition_family_error(self):
        mesh = self.mesh(2)
        evalues = Function(VectorFunctionSpace(mesh, "DG", 1))
        evectors = Function(TensorFunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            assemble_eigendecomposition(evectors, evalues)
        msg = (
            "Mismatching finite element families:"
            " 'Lagrange' vs. 'Discontinuous Lagrange'."
        )
        assert str(cm.exception) == msg

    def test_assemble_eigendecomposition_degree_error(self):
        mesh = self.mesh(2)
        evalues = Function(VectorFunctionSpace(mesh, "CG", 2))
        evectors = Function(TensorFunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            assemble_eigendecomposition(evectors, evalues)
        msg = "Mismatching finite element space degrees: 1 vs. 2."
        assert str(cm.exception) == msg

    @pytest.mark.slow
    @parameterized.expand([(2, True), (2, False), (3, True), (3, False)])
    def test_eigendecomposition(self, dim, reorder):
        """
        Check decomposition of a metric into its eigenvectors
        and eigenvalues.

          * The eigenvectors should be orthonormal.
          * Applying `compute_eigendecomposition` followed by
            `set_eigendecomposition` should get back the metric.
        """
        mesh = self.mesh(dim)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        # Create a simple metric
        metric = RiemannianMetric(P1_ten)
        mat = [[1, 0], [0, 2]] if dim == 2 else [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        metric.interpolate(as_matrix(mat))

        # Extract the eigendecomposition
        evectors, evalues = compute_eigendecomposition(metric, reorder=reorder)

        # Check eigenvectors are orthonormal
        err = Function(P1_ten)
        err.interpolate(dot(evectors, transpose(evectors)) - Identity(dim))
        if not np.isclose(norm(err), 0.0):
            raise ValueError(f"Eigenvectors are not orthonormal: {evectors.dat.data}")

        # Check eigenvalues are in descending order
        if reorder:
            P1 = FunctionSpace(mesh, "CG", 1)
            for i in range(dim - 1):
                f = interpolate(evalues[i], P1)
                f -= interpolate(evalues[i + 1], P1)
                if f.vector().gather().min() < 0.0:
                    raise ValueError(
                        f"Eigenvalues are not in descending order: {evalues.dat.data}"
                    )

        # Reassemble it and check the two match
        metric -= assemble_eigendecomposition(evectors, evalues)
        if not np.isclose(norm(metric), 0.0):
            raise ValueError(
                f"Reassembled metric does not match. Error: {metric.dat.data}"
            )

    @parameterized.expand([(2, True), (2, False), (3, True), (3, False)])
    def test_density_quotients_decomposition(self, dim, reorder):
        """
        Check decomposition of a metric into its density
        and anisotropy quotients.

        Reassembling should get back the metric.
        """
        mesh = self.mesh(dim)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        # Create a simple metric
        metric = RiemannianMetric(P1_ten)
        mat = [[1, 0], [0, 2]] if dim == 2 else [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        metric.interpolate(as_matrix(mat))

        # Extract the eigendecomposition
        evectors, evalues = compute_eigendecomposition(metric, reorder=reorder)

        # Check eigenvectors are orthonormal
        err = Function(P1_ten)
        err.interpolate(dot(evectors, transpose(evectors)) - Identity(dim))
        if not np.isclose(norm(err), 0.0):
            raise ValueError(f"Eigenvectors are not orthonormal: {evectors.dat.data}")

        # Check eigenvalues are in descending order
        if reorder:
            P1 = FunctionSpace(mesh, "CG", 1)
            for i in range(dim - 1):
                f = interpolate(evalues[i], P1)
                f -= interpolate(evalues[i + 1], P1)
                if f.vector().gather().min() < 0.0:
                    raise ValueError(
                        f"Eigenvalues are not in descending order: {evalues.dat.data}"
                    )

        # Reassemble it and check the two match
        metric -= assemble_eigendecomposition(evectors, evalues)
        if not np.isclose(norm(metric), 0.0):
            raise ValueError(
                f"Reassembled metric does not match. Error: {metric.dat.data}"
            )
