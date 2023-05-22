# Advection-diffusion-reaction with multiple prognostic variables
# ===============================================================
#
# In the the `previous demo <./gray_scott.py.html>`__, we solved the Gray-Scott
# equation using a mixed formulation for the two tracer species. Here, we instead use
# different fields for each of them, treating the corresponding equations separately.
# This considers an additional level of complexity compared with the
# `split solid body rotation demo <./solid_body_rotation_split.py.html>`__ because the
# equations differ in both the diffusion and reaction terms. ::

from firedrake import *
from pyroteus_adjoint import *

# This time, we have two fields instead of one, as well as two function spaces. ::

fields = ["a", "b"]
mesh = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction="both")


def get_function_spaces(mesh):
    return {
        "a": FunctionSpace(mesh, "CG", 1),
        "b": FunctionSpace(mesh, "CG", 1),
    }


# Therefore, the initial condition must be constructed using separate
# :class:`Function`\s. ::


def get_initial_condition(mesh_seq):
    x, y = SpatialCoordinate(mesh_seq[0])
    fs_a = mesh_seq.function_spaces["a"][0]
    fs_b = mesh_seq.function_spaces["b"][0]
    a_init = Function(fs_a, name="a")
    b_init = Function(fs_b, name="b")
    b_init.interpolate(
        conditional(
            And(And(1 <= x, x <= 1.5), And(1 <= y, y <= 1.5)),
            0.25 * sin(4 * pi * x) ** 2 * sin(4 * pi * y) ** 2,
            0,
        )
    )
    a_init.interpolate(1 - 2 * b_init)
    return {"a": a_init, "b": b_init}


# We now see why :func:`get_form` needs to provide a function whose return value is a
# dictionary: its keys correspond to the different equations being solved. ::


def get_form(mesh_seq):
    def form(index, sols):
        a, a_ = sols["a"]
        b, b_ = sols["b"]
        psi_a = TestFunction(mesh_seq.function_spaces["a"][index])
        psi_b = TestFunction(mesh_seq.function_spaces["b"][index])

        # Define constants
        dt = Constant(mesh_seq.time_partition[index].timestep)
        D_a = Constant(8.0e-05)
        D_b = Constant(4.0e-05)
        gamma = Constant(0.024)
        kappa = Constant(0.06)

        # Write the two equations in variational form
        F_a = (
            psi_a * (a - a_) * dx
            + dt * D_a * inner(grad(psi_a), grad(a)) * dx
            - dt * psi_a * (-a * b**2 + gamma * (1 - a)) * dx
        )
        F_b = (
            psi_b * (b - b_) * dx
            + dt * D_b * inner(grad(psi_b), grad(b)) * dx
            - dt * psi_b * (a * b**2 - (gamma + kappa) * b) * dx
        )
        return {"a": F_a, "b": F_b}

    return form


# Correspondingly the solver needs to be constructed from the two parts and must
# include two nonlinear solves at each timestep. ::


def get_solver(mesh_seq):
    def solver(index, ics):
        fs_a = mesh_seq.function_spaces["a"][index]
        fs_b = mesh_seq.function_spaces["b"][index]
        a = Function(fs_a, name="a")
        b = Function(fs_b, name="b")

        # Initialise 'lagged' solutions
        a_ = Function(fs_a, name="a_old")
        a_.assign(ics["a"])
        b_ = Function(fs_b, name="b_old")
        b_.assign(ics["b"])

        # Setup solver objects
        forms = mesh_seq.form(index, {"a": (a, a_), "b": (b, b_)})
        F_a = forms["a"]
        F_b = forms["b"]
        nlvp_a = NonlinearVariationalProblem(F_a, a)
        nlvs_a = NonlinearVariationalSolver(nlvp_a, ad_block_tag="a")
        nlvp_b = NonlinearVariationalProblem(F_b, b)
        nlvs_b = NonlinearVariationalSolver(nlvp_b, ad_block_tag="b")

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        while t < t_end - 0.5 * dt:
            nlvs_a.solve()
            nlvs_b.solve()
            a_.assign(a)
            b_.assign(b)
            t += dt
        return {"a": a, "b": b}

    return solver


# Let's consider the same QoI, time partition, and mesh sequence as in the previous
# demo, so that the outputs can be straightforwardly compared. ::


def get_qoi(mesh_seq, sols, index):
    def qoi():
        a = sols["a"]
        b = sols["b"]
        return a * b**2 * dx

    return qoi


test = os.environ.get("PYROTEUS_REGRESSION_TEST") is not None
end_time = 10.0 if test else 2000.0
dt = [0.0001, 0.001, 0.01, 0.1, (end_time - 1) / end_time]
num_subintervals = 5
dt_per_export = [10, 9, 9, 9, 10]
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    timesteps_per_export=dt_per_export,
    subintervals=[
        (0.0, 0.001),
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 1.0),
        (1.0, end_time),
    ],
)

mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

if not test:
    ic = mesh_seq.get_initial_condition()
    for field, sols in solutions.items():
        fwd_outfile = File(f"gray_scott_split/{field}_forward.pvd")
        adj_outfile = File(f"gray_scott_split/{field}_adjoint.pvd")
        fwd_outfile.write(ic[field])
        for i, mesh in enumerate(mesh_seq):
            for sol in sols["forward"][i]:
                fwd_outfile.write(sol)
            for sol in sols["adjoint"][i]:
                adj_outfile.write(sol)
        adj_outfile.write(Function(ic[field]).assign(0.0))

# This tutorial can be dowloaded as a `Python script <gray_scott_split.py>`__.
