"""
Test adjoint drivers.
"""
from pyroteus_adjoint import *
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    """
    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    yield
    import pyadjoint
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
    import pyadjoint
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


def test_adjoint_same_mesh(problem, qoi_type, debug=False):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :arg problem: string denoting the test case of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    """
    import importlib
    from firedrake_adjoint import pyadjoint

    # Imports
    pyrint(f"\n--- Setting up {problem} test case with {qoi_type} QoI\n")
    test_case = importlib.import_module(problem)
    end_time = test_case.end_time
    if "solid_body_rotation" in problem:
        end_time /= 4  # Reduce testing time

    # Partition time interval and create MeshSeq
    time_partition = TimePartition(
        end_time, 1, test_case.dt, test_case.fields,
        timesteps_per_export=test_case.dt_per_export,
    )
    mesh_seq = AdjointMeshSeq(
        time_partition, test_case.mesh, test_case.get_function_spaces,
        test_case.get_initial_condition, test_case.get_solver,
        test_case.get_qoi, qoi_type=qoi_type,
    )

    # Solve forward and adjoint without solve_adjoint
    pyrint("\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    ic = mesh_seq.initial_condition
    controls = [pyadjoint.Control(value) for key, value in ic.items()]
    sols = mesh_seq.solver(0, ic)
    J = mesh_seq.J if qoi_type == 'time_integrated' else mesh_seq.qoi(sols)
    pyadjoint.compute_gradient(J, controls)  # FIXME: gradient w.r.t. mixed function not correct
    J_expected = float(J)

    # Get expected adjoint solutions and values
    adj_sols_expected = {}
    adj_values_expected = {}
    for field, fs in mesh_seq._fs.items():
        solve_blocks = mesh_seq.get_solve_blocks(field)
        fwd_old_idx = mesh_seq.get_lagged_dependency_index(field, 0, solve_blocks)[0]
        adj_sols_expected[field] = solve_blocks[0].adj_sol.copy(deepcopy=True)
        adj_values_expected[field] = Function(
            fs[0], val=solve_blocks[0]._dependencies[fwd_old_idx].adj_value
        )

    # Loop over having one or two subintervals
    for N in range(1, 3):
        pyrint(f"\n--- Solving the adjoint problem on {N} subinterval"
               + f"{'' if N == 1 else 's'} using pyroteus\n")

        # Solve forward and adjoint on each subinterval
        time_partition = TimePartition(
            end_time, N, test_case.dt, test_case.fields,
            timesteps_per_export=test_case.dt_per_export,
        )
        mesh_seq = AdjointMeshSeq(
            time_partition, test_case.mesh, test_case.get_function_spaces,
            test_case.get_initial_condition, test_case.get_solver,
            test_case.get_qoi, qoi_type=qoi_type,
        )
        solutions = mesh_seq.solve_adjoint(get_adj_values=True, test_checkpoint_qoi=True)

        # Check quantities of interest match
        assert np.isclose(J_expected, mesh_seq.J), f"QoIs do not match ({J_expected} vs." \
                                                   + f"{mesh_seq.J})"

        # Check adjoint solutions at initial time match
        for field in time_partition.fields:
            adj_sol_expected = adj_sols_expected[field]
            adj_sol_computed = solutions[field].adjoint[0][0]
            err = errornorm(adj_sol_expected, adj_sol_computed)/norm(adj_sol_expected)
            assert np.isclose(err, 0.0), "Adjoint solutions at initial time do not match." \
                                         + f" (Error {err:.4e}.)"

        # Check adjoint actions at initial time match
        for field in time_partition.fields:
            adj_value_expected = adj_values_expected[field]
            adj_value_computed = solutions[field].adj_value[0][0]
            err = errornorm(adj_value_expected, adj_value_computed)/norm(adj_value_expected)
            assert np.isclose(err, 0.0), "Adjoint values at initial time do not match." \
                                         + f" (Error {err:.4e}.)"


@pytest.mark.parallel
def test_adjoint_same_mesh_parallel(problem, qoi_type):
    test_adjoint_same_mesh(problem, qoi_type)


if __name__ == "__main__":
    test_adjoint_same_mesh("migrating_trench", "end_time", debug=True)
