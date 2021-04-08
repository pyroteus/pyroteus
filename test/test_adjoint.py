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

@pytest.fixture(params=["burgers"])
def pde(request):
    return request.param


@pytest.fixture(params=["end_time", "time_integrated"])
def qoi_type(request):
    return request.param


def test_adjoint_same_mesh(pde, qoi_type, plot=False):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :arg pde: string denoting the PDE of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    :kwarg plot: toggle plotting of the adjoint
        solution field
    """
    from firedrake_adjoint import Control

    # Setup
    if pde == "burgers":
        import burgers as PDE
    else:
        raise NotImplementedError  # TODO: test solve_adjoint for mixed spaces
    fs, end_time, dt = PDE.setup()
    qoi = PDE.end_time_qoi if qoi_type == 'end_time' else PDE.time_integrated_qoi

    # Solve forward and adjoint without the subinterval framework
    solver_kwargs = {}
    if qoi_type == 'time_integrated':
        solver_kwargs['qoi'] = lambda *args: assemble(qoi(*args))
    ic = PDE.initial_condition(fs)
    sol, J = PDE.solver(ic, 0.0, end_time, dt, **solver_kwargs)
    if qoi_type == 'end_time':
        J = assemble(qoi(sol))
    pyadjoint.compute_gradient(J, Control(ic))
    tape = pyadjoint.get_working_tape()
    solve_blocks = [
        block for block in tape.get_blocks()
        if issubclass(block.__class__, GenericSolveBlock)
        and block.adj_sol is not None
    ]
    final_adj_sols = [solve_blocks[0].adj_sol]
    qois = [J]

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        dt_per_export = 2
        N = len(spaces)
        subintervals = get_subintervals(end_time, N)

        # Solve forward and adjoint on each subinterval
        J, adj_sols = solve_adjoint(
            PDE.solver, PDE.initial_condition, qoi, spaces, end_time, dt,
            timesteps_per_export=dt_per_export
        )
        final_adj_sols.append(adj_sols[-1][-1])
        qois.append(J)
        assert norm(adj_sols[-1][-1]) > 1.0e-08

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
            plt.savefig(f"plots/burgers_test_{N}_{qoi_type}.jpg")

        # Check adjoint solutions at initial time match
        assert np.isclose(errornorm(*final_adj_sols[::N])/norm(final_adj_sols[0]), 0.0)

        # Check quantities of interest match
        assert np.isclose(*qois[::N])


# ---------------------------
# plotting and debugging
# ---------------------------

if __name__ == "__main__":
    test_adjoint_same_mesh("burgers", "end_time", plot=True)
    test_adjoint_same_mesh("burgers", "time_integrated", plot=True)
