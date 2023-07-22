"""
Testing for the mesh sequence objects.
"""
from pyroteus.mesh_seq import MeshSeq
from pyroteus.time_partition import TimePartition, TimeInterval
from firedrake import UnitCubeMesh, UnitSquareMesh, UnitTriangleMesh
import numpy as np
import re
import unittest


class TestGeneric(unittest.TestCase):
    """
    Generic unit tests for :class:`MeshSeq`.
    """

    def setUp(self):
        self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])

    def test_setitem(self):
        mesh1 = UnitSquareMesh(1, 1, diagonal="left")
        mesh2 = UnitSquareMesh(1, 1, diagonal="right")
        mesh_seq = MeshSeq(self.time_interval, [mesh1])
        self.assertEqual(mesh_seq[0], mesh1)
        mesh_seq[0] = mesh2
        self.assertEqual(mesh_seq[0], mesh2)

    def test_inconsistent_dim(self):
        meshes = [UnitSquareMesh(1, 1), UnitCubeMesh(1, 1, 1)]
        with self.assertRaises(ValueError) as cm:
            MeshSeq(self.time_partition, meshes)
        msg = "Meshes must all have the same topological dimension."
        self.assertEqual(str(cm.exception), msg)

    def test_get_function_spaces_notimplemented_error(self):
        meshes = [UnitSquareMesh(1, 1)]
        mesh_seq = MeshSeq(self.time_interval, meshes)
        with self.assertRaises(NotImplementedError) as cm:
            mesh_seq.get_function_spaces(meshes[0])
        msg = "'get_function_spaces' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    def test_get_form_notimplemented_error(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        with self.assertRaises(NotImplementedError) as cm:
            mesh_seq.get_form()
        msg = "'get_form' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    def test_get_solver_notimplemented_error(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        with self.assertRaises(NotImplementedError) as cm:
            mesh_seq.get_solver()
        msg = "'get_solver' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    def test_element_convergence_lt_miniter(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitTriangleMesh()])
        mesh_seq.check_element_count_convergence()
        self.assertFalse(mesh_seq.converged)

    def test_element_convergence_true(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitTriangleMesh()])
        mesh_seq.element_counts = np.ones((mesh_seq.params.miniter + 1, 1))
        mesh_seq.check_element_count_convergence()
        self.assertTrue(mesh_seq.converged)

    def test_element_convergence_false(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        mesh_seq.element_counts = np.ones((mesh_seq.params.miniter + 1, 1))
        mesh_seq.element_counts[-1] = 2
        mesh_seq.check_element_count_convergence()
        self.assertFalse(mesh_seq.converged)

    def test_counting(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [18])
        self.assertEqual(mesh_seq.count_vertices(), [16])


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__` and :meth:`__repr__` methods work as intended for
    Pyroteus' :class:`MeshSeq` object.
    """

    def setUp(self):
        self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])

    def test_mesh_seq_time_interval_str(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>']")

    def test_mesh_seq_time_partition_str(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>', '<Mesh ?>']")

    def test_mesh_seq_time_interval_repr(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        expected = "MeshSeq([Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_time_partition_2_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_time_partition_3_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
            UnitSquareMesh(1, 1, diagonal="left"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "..."
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))
