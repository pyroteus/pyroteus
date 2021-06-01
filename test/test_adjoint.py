"""
Test adjoint drivers.
"""
from pyroteus_adjoint import *
from pyroteus.adjoint import count_qoi_args
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
def handle_exit_annotation():
    """
    Since importing firedrake_adjoint modifies a global variable, we need to
    pause annotations at the end of the module.

    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    yield
    annotate = pyadjoint.annotate_tape()
    if annotate:
        pyadjoint.pause_annotation()


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[
    "burgers",
    "solid_body_rotation",
    "solid_body_rotation_split",
    "rossby_wave",
    "migrating_trench",
])
def problem(request):
    return request.param


@pytest.fixture(params=[
    "end_time",
    "time_integrated",
])
def qoi_type(request):
    return request.param


def test_adjoint_same_mesh(problem, qoi_type):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :arg problem: string denoting the test case of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    """
    import importlib
    from firedrake_adjoint import Control

    if problem == "migrating_trench":
        pytest.xfail("FIXME: trench test not correctly annotated")  # FIXME

    # Setup
    print(f"\n--- Setting up {problem} test case with {qoi_type} QoI\n")
    test_case = importlib.import_module(problem)
    fs = test_case.function_space
    end_time = test_case.end_time
    if "solid_body_rotation" in problem:
        end_time /= 4  # Reduce testing time
    qoi = test_case.end_time_qoi if qoi_type == 'end_time' else test_case.time_integrated_qoi
    assert count_qoi_args(qoi) == qoi_type, "Inconsistent QoI type"
    time_partition = TimePartition(
        end_time, 1, test_case.dt, test_case.fields,
        timesteps_per_export=test_case.dt_per_export,
        solves_per_timestep=test_case.solves_per_dt,
    )
    fields = time_partition.fields

    # Solve forward and adjoint without solve_adjoint
    print("\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    solver_kwargs = {}
    if qoi_type == 'time_integrated':
        solver_kwargs['qoi'] = lambda *args: assemble(qoi(*args))
    ic = test_case.initial_condition({field: [fs[field]] for field in fields})
    controls = [Control(value) for key, value in ic.items()]
    sols, J = test_case.solver(ic, 0.0, end_time, test_case.dt, **solver_kwargs)
    if qoi_type == 'end_time':
        J = assemble(qoi(sols))
    pyadjoint.compute_gradient(J, controls)  # FIXME: gradient w.r.t. mixed function not correct
    J_expected = float(J)
    adj_sols_expected = {
        field: time_partition.get_solve_blocks(field)[0].adj_sol.copy(deepcopy=True)
        for field in fields
    }

    # Loop over having one or two subintervals
    for N in range(1, 3):
        spaces = {field: N*[fs[field]] for field in fields}
        print(f"\n--- Solving the adjoint problem on {N} subinterval"
              + f"{'' if N == 1 else 's'} using pyroteus\n")

        # Solve forward and adjoint on each subinterval
        time_partition = TimePartition(
            end_time, N, test_case.dt, test_case.fields,
            timesteps_per_export=test_case.dt_per_export,
            solves_per_timestep=test_case.solves_per_dt,
        )
        J, solutions = solve_adjoint(
            test_case.solver, test_case.initial_condition, qoi, spaces, time_partition,
        )

        # Check quantities of interest match
        assert np.isclose(J_expected, J), f"QoIs do not match ({J_expected} vs. {J})"

        # Check adjoint solutions at initial time match
        for field in time_partition.fields:
            adj_sol_expected = adj_sols_expected[field]
            adj_sol_computed = solutions[field].adjoint[0][0]
            err = errornorm(adj_sol_expected, adj_sol_computed)/norm(adj_sol_expected)
            assert np.isclose(err, 0.0), f"Non-zero adjoint error ({err})"


@pytest.mark.parallel
def test_adjoint_same_mesh_parallel(problem, qoi_type):
    test_adjoint_same_mesh(problem, qoi_type)


if __name__ == "__main__":
    test_adjoint_same_mesh("migrating_trench", "end_time")
