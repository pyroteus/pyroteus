import pyadjoint
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    """
    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    yield
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    """
    Since importing firedrake-adjoint modifies a global variable, we need to
    pause annotations at the end of the module.

    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    if not pyadjoint.annotate_tape():
        pyadjoint.continue_annotation()
    yield
    if pyadjoint.annotate_tape():
        pyadjoint.pause_annotation()
