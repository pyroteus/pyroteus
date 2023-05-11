"""
Testing for the mesh sequence objects.
"""
from pyroteus.mesh_seq import MeshSeq
from pyroteus.time_partition import TimePartition, TimeInterval
from firedrake import UnitSquareMesh
import re
import unittest


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
