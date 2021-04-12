"""
Test interpolation schemes.

**Disclaimer: `test_project_parallel` and its is associated fixtures are
    copied from firedrake/tests/supermesh/test_nonnested_project_no_hierarchy.py
"""
from pyroteus import *
from itertools import product
import pytest
import weakref


spaces = [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)]


@pytest.fixture(params=[(c, f) for c, f in product(spaces, spaces) if c[1] <= f[1]],
                ids=lambda x: "%s%s-%s%s" % (*x[0], *x[1]))
def pairs(request):
    return request.param


@pytest.fixture
def coarse(pairs):
    return pairs[0]


@pytest.fixture
def fine(pairs):
    return pairs[1]


def test_project(coarse, fine):
    """
    **Disclaimer: this test and its is associated fixtures are copied from
        firedrake/tests/supermesh/test_nonnested_project_no_hierarchy.py
    """
    distribution_parameters = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 10),
    }
    cmesh = RectangleMesh(2, 2, 1, 1, diagonal="left",
                          distribution_parameters=distribution_parameters)
    fmesh = RectangleMesh(5, 5, 1, 1, diagonal="right",
                          distribution_parameters=distribution_parameters)
    fmesh._parallel_compatible = {weakref.ref(cmesh)}

    Vc = FunctionSpace(cmesh, *coarse)
    Vf = FunctionSpace(fmesh, *fine)

    c = Function(Vc)
    c.interpolate(SpatialCoordinate(cmesh)**2)
    expect = assemble(c*dx)

    actual = assemble(mesh2mesh_project(c, Vf)*dx)

    assert np.isclose(expect, actual)


@pytest.mark.parallel(nprocs=3)
def test_project_parallel(coarse, fine):
    test_project(coarse, fine)
