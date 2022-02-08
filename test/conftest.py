"""
Global pytest configuration.

**Disclaimer: some functions copied from firedrake/src/tests/conftest.py
"""
import pyadjoint
import pytest
from subprocess import check_call


def parallel(item):
    """
    Run a test in parallel.

    **Disclaimer: copied from firedrake/src/tests/conftest.py

    :arg item: the test item to run.
    """
    from mpi4py import MPI

    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("Parallel test can't be run within parallel environment")
    marker = item.get_closest_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process.
    call = [
        "mpiexec",
        "-n",
        "1",
        "python",
        "-m",
        "pytest",
        "--runxfail",
        "-s",
        "-q",
        "%s::%s" % (item.fspath, item.name),
    ]
    call.extend(
        [
            ":",
            "-n",
            "%d" % (nprocs - 1),
            "python",
            "-m",
            "pytest",
            "--runxfail",
            "--tb=no",
            "-q",
            "%s::%s" % (item.fspath, item.name),
        ]
    )
    check_call(call)


def pytest_configure(config):
    """
    Register an additional marker.

    **Disclaimer: copied from firedrake/src/tests/conftest.py
    """
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors",
    )


def pytest_runtest_setup(item):
    """
    **Disclaimer: copied from firedrake/src/tests/conftest.py
    """
    if item.get_closest_marker("parallel"):
        from mpi4py import MPI

        if MPI.COMM_WORLD.size > 1:
            # Turn on source hash checking
            from firedrake import parameters
            from functools import partial

            def _reset(check):
                parameters["pyop2_options"]["check_src_hashes"] = check

            # Reset to current value when test is cleaned up
            item.addfinalizer(
                partial(_reset, parameters["pyop2_options"]["check_src_hashes"])
            )

            parameters["pyop2_options"]["check_src_hashes"] = True
        else:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: True


def pytest_runtest_call(item):
    """
    **Disclaimer: copied from firedrake/src/tests/conftest.py
    """
    from mpi4py import MPI

    if item.get_closest_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """
    Check that the tape is empty at the end of each module

    **Disclaimer: copied from firedrake/src/tests/conftest.py
    """

    def fin():
        tape = pyadjoint.get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)


def pytest_runtest_teardown(item, nextitem):
    """
    Clear caches after running a test
    """
    from firedrake.tsfc_interface import TSFCKernel
    from pyop2.op2 import Kernel
    from pyop2.parloop import JITModule

    Kernel._cache.clear()
    TSFCKernel._cache.clear()
    JITModule._cache.clear()
