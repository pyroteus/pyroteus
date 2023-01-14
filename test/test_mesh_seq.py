"""
Testing for the mesh sequence objects.
"""
from pyroteus.mesh_seq import MeshSeq
from pyroteus.time_partition import TimePartition, TimeInterval
from firedrake import UnitSquareMesh
import unittest


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__` and :meth:`__repr__` methods work as intended for
    Pyroteus' :class:`MeshSeq` object.
    """

    _index = 0

    @property
    def index(self):
        TestStringFormatting._index += 1
        return TestStringFormatting._index

    def setUp(self):
        self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])

    def test_mesh_seq_time_interval_str(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        expected = f"['<Mesh #{self.index}>']"
        assert str(mesh_seq) == expected

    def test_mesh_seq_time_partition_str(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = f"['<Mesh #{self.index}>', '<Mesh #{self.index}>']"
        assert str(mesh_seq) == expected

    def test_mesh_seq_time_interval_repr(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        expected = f"MeshSeq([Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), {self.index})])"
        assert repr(mesh_seq) == expected

    def test_mesh_seq_time_partition_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = (
            "MeshSeq(["
            f"Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), {self.index}), "
            f"Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), {self.index})])"
        )
        assert repr(mesh_seq) == expected
