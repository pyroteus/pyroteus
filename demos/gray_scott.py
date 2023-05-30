# Advection-diffusion-reaction
# ============================
#
# We have already seen time-dependent advection equations and a steady-state
# advection-diffusion equation. In this demo, we increase the level of complexity again
# and solve the Gray-Scott advection-diffusion-reaction equation, as described in
# :cite:`Hundsdorfer:2003`.
#
# The test case consists of two tracer species, which experience different
# diffusivities and react with one another nonlinearly. ::

from firedrake import *
from pyroteus_adjoint import *

# The problem is defined on a doubly periodic mesh of squares. ::

fields = ["ab"]
mesh = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction="both")

# We solve for the tracer species using a mixed formulation, with a :math:`\mathbb P1`
# approximation for both components. ::


def get_function_spaces(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    return {"ab": V * V}


# The initial conditions are localised within the region :math:`[1, 1.5]^2`. ::


def get_initial_condition(mesh_seq):
    x, y = SpatialCoordinate(mesh_seq[0])
    fs = mesh_seq.function_spaces["ab"][0]
    ab_init = Function(fs)
    a_init, b_init = ab_init.subfunctions
    b_init.interpolate(
        conditional(
            And(And(1 <= x, x <= 1.5), And(1 <= y, y <= 1.5)),
            0.25 * sin(4 * pi * x) ** 2 * sin(4 * pi * y) ** 2,
            0,
        )
    )
    a_init.interpolate(1 - 2 * b_init)
    return {"ab": ab_init}


# Since we are using a mixed formulation, the forms for each component equation are
# summed together. ::


def get_form(mesh_seq):
    def form(index, sols):
        ab, ab_ = sols["ab"]
        a, b = split(ab)
        a_, b_ = split(ab_)
        psi_a, psi_b = TestFunctions(mesh_seq.function_spaces["ab"][index])

        # Define constants
        dt = Constant(mesh_seq.time_partition[index].timestep)
        D_a = Constant(8.0e-05)
        D_b = Constant(4.0e-05)
        gamma = Constant(0.024)
        kappa = Constant(0.06)

        # Write the two equations in variational form
        F = (
            psi_a * (a - a_) * dx
            + dt * D_a * inner(grad(psi_a), grad(a)) * dx
            - dt * psi_a * (-a * b**2 + gamma * (1 - a)) * dx
            + psi_b * (b - b_) * dx
            + dt * D_b * inner(grad(psi_b), grad(b)) * dx
            - dt * psi_b * (a * b**2 - (gamma + kappa) * b) * dx
        )
        return {"ab": F}

    return form


# For the solver, we just use the default configuration of Firedrake's
# :class:`NonlinearVariationalSolver`. ::


def get_solver(mesh_seq):
    def solver(index, ics):
        fs = mesh_seq.function_spaces["ab"][index]
        ab = Function(fs, name="ab")

        # Initialise 'lagged' solution
        ab_ = Function(fs, name="ab_old")
        ab_.assign(ics["ab"])

        # Setup solver objects
        F = mesh_seq.form(index, {"ab": (ab, ab_)})["ab"]
        nlvp = NonlinearVariationalProblem(F, ab)
        nlvs = NonlinearVariationalSolver(nlvp, ad_block_tag="ab")

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        while t < t_end - 0.5 * dt:
            nlvs.solve()
            ab_.assign(ab)
            t += dt
        return {"ab": ab}

    return solver


# The term :math:`a * b ^ 2` appears in both equations. By solving the adjoint for the
# QoI :math:`\int a(x,T) * b(x,T) * dx` we consider sensitivities to this term. ::


def get_qoi(mesh_seq, sols, index):
    def qoi():
        ab = sols["ab"]
        a, b = split(ab)
        return a * b**2 * dx

    return qoi


# This problem is multi-scale in time and requires spinning up by gradually increasing
# the timestep. This is straightforwardly done in Pyroteus using :class:`TimePartition`.
# ::

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

# As usual, define an appropriate :class:`MeshSeq` and choose the `qoi_type`. ::

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

# Finally, plot the outputs to be viewed in Paraview. ::

if not test:
    ic = mesh_seq.get_initial_condition()
    for field, sols in solutions.items():
        fwd_outfile = File(f"gray_scott/{field}_forward.pvd")
        adj_outfile = File(f"gray_scott/{field}_adjoint.pvd")
        fwd_outfile.write(*ic[field].subfunctions)
        for i, mesh in enumerate(mesh_seq):
            for sol in sols["forward"][i]:
                fwd_outfile.write(*sol.subfunctions)
            for sol in sols["adjoint"][i]:
                adj_outfile.write(*sol.subfunctions)
        adj_end = Function(ic[field]).assign(0.0)
        adj_outfile.write(*adj_end.subfunctions)

# In the `next demo <./gray_scott_split.py.html>`__, we consider solving the same
# problem, but splitting the solution field into multiple components.
#
# This tutorial can be dowloaded as a `Python script <gray_scott.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
