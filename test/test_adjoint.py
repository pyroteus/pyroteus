from pyroteus import *
from pyadjoint.tape import get_working_tape, pause_annotation, annotate_tape
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    """
    **Disclaimer: copied from firedrake/tests/regression/test_adjoint_interpolate.py
    """
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_exit_annotation():
    """
    Since importing firedrake_adjoint modifies a global variable, we need to
    pause annotations at the end of the module.

    **Disclaimer: copied from firedrake/tests/regression/test_adjoint_interpolate.py
    """
    yield
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


# TODO: test solve_adjoint for mixed spaces

def solve_burgers(ic, t_start, t_end, dt, J=0, qoi=None, **kwargs):
    """
    Solve Burgers' equation on an interval
    (t_start, t_end), given viscosity `nu` and some
    initial condition `ic`.
    """
    fs = ic.function_space()
    dtc = Constant(dt)

    # Check viscosity parameter gets passed through
    nu = kwargs.get('viscosity')
    if nu is None:
        raise ValueError

    # Set initial condition
    u_ = Function(fs)
    u_.assign(ic)

    # Setup variational problem
    v = TestFunction(fs)
    u = Function(fs)
    F = inner((u - u_)/dtc, v)*dx \
        + inner(dot(u, nabla_grad(u)), v)*dx \
        + nu*inner(grad(u), grad(v))*dx

    # Time integrate from t_start to t_end
    t = t_start
    while t < t_end - 1.0e-05:
        solve(F == 0, u)
        if qoi is not None:
            J += qoi(u, t)
        u_.assign(u)
        t += dt
    return u_, J


def initial_condition_burgers(fs):
    """
    Initial condition for Burgers' equation
    which is sinusoidal in the x-direction.

    :arg fs: :class:`FunctionSpace` which
        the initial condition will live in
    """
    x, y = SpatialCoordinate(fs.mesh())
    return interpolate(as_vector([sin(pi*x), 0]), fs)


# TODO: test solve_adjoint for time-integrated QoIs

def end_time_qoi_burgers(sol):
    """
    Quantity of interest for Burgers' equation
    which computes the square L2 norm over the
    right hand boundary segment.

    :arg sol: the solution :class:`Function`
    """
    return inner(sol, sol)*ds(2)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[end_time_qoi_burgers])
def qoi(request):
    return request.param


def test_adjoint_burgers_same_mesh(qoi, plot=False):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :kwarg plot: toggle plotting of the adjoint
        solution field
    """
    import firedrake_adjoint  # noqa
    n = 32
    mesh = UnitSquareMesh(n, n, diagonal='left')
    end_time = 0.5
    dt = 1/n

    # Loop over having one or two subintervals
    final_adj_sols = []
    qois = []
    for meshes in ([mesh], [mesh, mesh]):
        dt_per_export = 2
        N = len(meshes)
        subintervals = get_subintervals(end_time, N)

        # Solve forward and adjoint on each subinterval
        spaces = [VectorFunctionSpace(mesh, "CG", 2) for mesh in meshes]
        J, adj_sols = solve_adjoint(
            solve_burgers, initial_condition_burgers, qoi, spaces, end_time, dt,
            timesteps_per_export=dt_per_export, solver_kwargs=dict(viscosity=Constant(0.0001)),
        )
        final_adj_sols.append(adj_sols[-1][-1])
        qois.append(J)
        assert norm(adj_sols[-1][-1]) > 1.0e-08

        # Plot adjoint solutions, if requested
        if plot:
            import matplotlib.pyplot as plt
            from pyroteus.ts import get_exports_per_subinterval

            levels = np.linspace(0, 0.8, 9)
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
            plt.savefig(f"plots/burgers_test_{N}.jpg")

    # Check adjoint solutions at initial time match
    assert np.isclose(errornorm(*final_adj_sols)/norm(final_adj_sols[0]), 0.0)

    # Check quantities of interest match
    assert np.isclose(*qois)


# ---------------------------
# plotting and debugging
# ---------------------------

if __name__ == "__main__":
    test_adjoint_burgers(plot=True)
