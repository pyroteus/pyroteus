"""
Test adjoint drivers.
"""
from pyroteus import *
from firedrake.adjoint.blocks import GenericSolveBlock
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


def test_adjoint_same_mesh(problem, qoi_type, plot=False):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :arg problem: string denoting the test case of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    :kwarg plot: toggle plotting of the adjoint
        solution field
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
        pytest.xfail("FIXME")  # FIXME: invalid type conversion error
    elif problem == "rossby_wave" and qoi_type == "time_integrated":
        pytest.xfail("FIXME")  # FIXME: adjoint solutions don't match
    dt = test_case.dt
    dt_per_export = test_case.dt_per_export
    solves_per_dt = test_case.solves_per_dt
    qoi = test_case.end_time_qoi if qoi_type == 'end_time' else test_case.time_integrated_qoi
    num_timesteps = int(end_time/dt)

    # Solve forward and adjoint without the subinterval framework
    print(f"\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    solver_kwargs = {}
    if qoi_type == 'time_integrated':
        solver_kwargs['qoi'] = lambda *args: assemble(qoi(*args))
    ic = test_case.initial_condition(fs)
    sol, J = test_case.solver(ic, 0.0, end_time, dt, **solver_kwargs)
    if qoi_type == 'end_time':
        J = assemble(qoi(sol))
    pyadjoint.compute_gradient(J, Control(ic))
    tape = pyadjoint.get_working_tape()
    solve_blocks = [
        block for block in tape.get_blocks()
        if issubclass(block.__class__, GenericSolveBlock)
        and block.adj_sol is not None
    ][-num_timesteps*solves_per_dt::solves_per_dt]
    final_adj_sols = [solve_blocks[0].adj_sol]
    qois = [J]

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        N = len(spaces)
        plural = '' if N == 1 else 's'
        print(f"\n--- Solving the adjoint problem on {N} subinterval{plural} using pyroteus\n")
        subintervals = get_subintervals(end_time, N)

        # Solve forward and adjoint on each subinterval
        J, adj_sols = solve_adjoint(
            test_case.solver, test_case.initial_condition, qoi, spaces, end_time, dt,
            timesteps_per_export=dt_per_export, solves_per_timestep=solves_per_dt,
        )
        final_adj_sols.append(adj_sols[-1][-1])
        qois.append(J)

        # Plot adjoint solutions, if requested
        if plot:
            import matplotlib.pyplot as plt
            from pyroteus.ts import get_exports_per_subinterval

            levels = np.linspace(0, 0.8, 9) if qoi_type == 'end_time' else 9

            _, dt_per_export, exports_per_mesh = \
                get_exports_per_subinterval(subintervals, dt, dt_per_export)
            fig, axes = plt.subplots(exports_per_mesh[0], N, sharex='col', figsize=(6*N, 24//N))
            for i, adj_sols_step in enumerate(adj_sols):
                ax = axes[0] if N == 1 else axes[0, i]
                ax.set_title("Mesh {:d}".format(i+1))
                for j, adj_sol in enumerate(adj_sols_step):
                    ax = axes[j] if N == 1 else axes[j, i]
                    tricontourf(adj_sol, axes=ax, levels=levels)
                    ax.annotate(
                        f"t={(N-i)*end_time/N - j*dt_per_export[0]*dt:.2f}",
                        (0.05, 0.05),
                        color='white',
                    )
            plt.savefig(f"plots/{problem}_test_{N}_{qoi_type}.jpg")

        # Check quantities of interest match
        assert np.isclose(*qois[::N]), f"QoIs do not match ({qois[0]} vs. {qois[-1]})"

        # Check adjoint solutions at initial time match
        err = errornorm(*final_adj_sols[::N])/norm(final_adj_sols[0])
        assert np.isclose(err, 0.0), f"Non-zero adjoint error ({err})"


# ---------------------------
# plotting and debugging
# ---------------------------

if __name__ == "__main__":
    # test_adjoint_same_mesh("burgers", "end_time", plot=True)
    # test_adjoint_same_mesh("burgers", "time_integrated", plot=True)
    # test_adjoint_same_mesh("solid_body_rotation", "end_time", plot=False)
    # test_adjoint_same_mesh("solid_body_rotation", "time_integrated", plot=False)
    # test_adjoint_same_mesh("rossby_wave", "end_time", plot=False)
    test_adjoint_same_mesh("rossby_wave", "time_integrated", plot=False)
