"""
Test adjoint drivers.
"""
from pyroteus_adjoint import *
import pyadjoint
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    """
    **Disclaimer: copied from firedrake/tests/regression/test_adjoint_interpolate.py
    """
    yield
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_exit_annotation():
    """
    Since importing firedrake_adjoint modifies a global variable, we need to
    pause annotations at the end of the module.

    **Disclaimer: copied from firedrake/tests/regression/test_adjoint_interpolate.py
    """
    yield
    annotate = pyadjoint.annotate_tape()
    if annotate:
        pyadjoint.pause_annotation()


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=["burgers", "solid_body_rotation", "rossby_wave"])
def problem(request):
    return request.param


@pytest.fixture(params=["end_time", "time_integrated"])
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

    # Setup
    print(f"\n--- Setting up {problem} test case with {qoi_type} QoI\n")
    test_case = importlib.import_module(problem)
    fs = test_case.function_space
    end_time = test_case.end_time
    if problem == "solid_body_rotation" and qoi_type == "time_integrated":
        end_time /= 4  # Reduce testing time
    qoi = test_case.end_time_qoi if qoi_type == 'end_time' else test_case.time_integrated_qoi
    time_partition = TimePartition(
        end_time, 1, test_case.dt, timesteps_per_export=test_case.dt_per_export,
        solves_per_timestep=test_case.solves_per_dt,
    )

    # Solve forward and adjoint without solve_adjoint
    print("\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    solver_kwargs = {}
    if qoi_type == 'time_integrated':
        solver_kwargs['qoi'] = lambda *args: assemble(qoi(*args))
    ic = test_case.initial_condition(fs)
    sol, J = test_case.solver(ic, 0.0, end_time, test_case.dt, **solver_kwargs)
    if qoi_type == 'end_time':
        J = assemble(qoi(sol))
    pyadjoint.compute_gradient(J, Control(ic))  # FIXME: gradient w.r.t. mixed function not correct
    _adj_sol = time_partition.solve_blocks()[0].adj_sol.copy(deepcopy=True)
    _J = float(J)

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        N = len(spaces)
        print(f"\n--- Solving the adjoint problem on {N} subinterval"
              + f"{'' if N == 1 else 's'} using pyroteus\n")

        # Solve forward and adjoint on each subinterval
        time_partition = TimePartition(
            end_time, N, test_case.dt, timesteps_per_export=test_case.dt_per_export,
            solves_per_timestep=test_case.solves_per_dt,
        )
        J, solutions = solve_adjoint(
            test_case.solver, test_case.initial_condition, qoi, spaces, time_partition,
        )

        # Check quantities of interest match
        assert np.isclose(_J, J), f"QoIs do not match ({_J} vs. {J})"

        # Check adjoint solutions at initial time match
        err = errornorm(_adj_sol, solutions['adjoint'][0][0])/norm(_adj_sol)
        assert np.isclose(err, 0.0), f"Non-zero adjoint error ({err})"


# FIXME
# @pytest.mark.parallel
# def test_adjoint_same_mesh_parallel(problem, qoi_type):
#     test_adjoint_same_mesh(problem, qoi_type)


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    import argparse
    import burgers
    import matplotlib.pyplot as plt

    # Parse arguments
    parser = argparse.ArgumentParser(prog='pyroteus/adjoint.py')
    parser.add_argument('-qoi_type', help="Choose from 'end_time' or 'time_integrated'")
    _qoi_type = parser.parse_args().qoi_type or 'end_time'

    # Setup Burgers test case
    fs = burgers.function_space
    end_time = burgers.end_time
    qoi = burgers.end_time_qoi if _qoi_type == 'end_time' else burgers.time_integrated_qoi

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        N = len(spaces)
        print(f"\n--- Solving the adjoint problem on {N} subinterval{'' if N == 1 else 's'}\n")

        # Solve forward and adjoint on each subinterval
        P = TimePartition(
            end_time, N, burgers.dt, timesteps_per_export=burgers.dt_per_export, debug=True,
        )
        J, solutions = solve_adjoint(
            burgers.solver, burgers.initial_condition, qoi, spaces, P,
        )

        # Plot adjoint solutions
        Nx = P.exports_per_subinterval[0]-1
        fig, axes = plt.subplots(Nx, N, sharex='col', figsize=(6*N, 24//N))
        levels = np.linspace(0, 0.8, 9) if _qoi_type == 'end_time' else 9
        for i, adj_sols_step in enumerate(solutions['adjoint']):
            ax = axes[0] if N == 1 else axes[0, i]
            ax.set_title(f"Mesh {i+1}")
            for j, adj_sol in enumerate(adj_sols_step):
                ax = axes[j] if N == 1 else axes[j, i]
                tricontour(adj_sol, axes=ax, levels=levels)
                ax.annotate(
                    f"t={i*end_time/N + j*P.timesteps_per_export[i]*P.timesteps[i]:.2f}",
                    (0.05, 0.05),
                    color='white',
                )
        plt.savefig(f"plots/burgers_test_{N}_{_qoi_type}.jpg")
