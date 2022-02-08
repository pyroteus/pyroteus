# Adjoint of Burgers equation with a time integrated QoI
# ======================================================
#
# So far, we only considered a quantity of interest
# corresponding to a spatial integral at the end time.
# For some problems, it is more suitable to have a QoI
# which integrates in time as well as space.
#
# Begin by importing from Pyroteus and the first Burgers
# demo. ::

from firedrake import *
from pyroteus_adjoint import *
from burgers import get_initial_condition, get_function_spaces

# The solver needs to be modified slightly in order to take
# account of time dependent QoIs. The Burgers solver
# uses backward Euler timestepping. The corresponding
# quadrature routine is like the midpoint rule, but takes
# the value from the next timestep, rather than the average
# between that and the current value. As such, the QoI may
# be computed by simply incrementing as follows. ::


def get_solver(mesh_seq):
    def solver(index, ic):
        t_start, t_end = mesh_seq.subintervals[index]
        dt = mesh_seq.time_partition.timesteps[index]
        function_space = mesh_seq.function_spaces["uv_2d"][index]

        # Specify constants
        dtc = Constant(dt)
        nu = Constant(0.0001)

        # Set initial condition
        u_ = Function(function_space, name="uv_2d_old")
        u_.assign(ic["uv_2d"])

        # Setup variational problem
        v = TestFunction(function_space)
        u = Function(function_space, name="uv_2d")
        F = (
            inner((u - u_) / dtc, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )

        # Time integrate from t_start to t_end
        t = t_start
        qoi = mesh_seq.get_qoi(index)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="uv_2d")
            mesh_seq.J += qoi({"uv_2d": u}, t)
            u_.assign(u)
            t += dt
        return {"uv_2d": u}

    return solver


# The QoI is effectively just a time-integrated version
# of the one previously seen.
#
# .. math::
#    J(u) = \int_0^{T_{\mathrm{end}}} \int_0^1
#    \mathbf u(1,y,t) \cdot \mathbf u(1,y,t)
#    \;\mathrm dy\;\mathrm dt.
#
# Note that in this case we multiply by the timestep.
# It is wrapped in a :class:`Constant` to avoid
# recompilation if the value is changed. ::


def get_qoi(mesh_seq, i):
    dtc = Constant(mesh_seq.time_partition[i].timestep)

    def time_integrated_qoi(sol, t):
        u = sol["uv_2d"]
        return dtc * inner(u, u) * ds(2)

    return time_integrated_qoi


# We use the same mesh setup as in `the previous demo
# <./burgers2.py.html>`__ and the same time partitioning. ::

fields = ["uv_2d"]
n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n

num_subintervals = 2
P = TimePartition(end_time, num_subintervals, dt, fields, timesteps_per_export=2)
mesh_seq = AdjointMeshSeq(
    P,
    meshes,
    get_function_spaces,
    get_initial_condition,
    get_solver,
    get_qoi,
    qoi_type="time_integrated",
)
solutions = mesh_seq.solve_adjoint()

# Finally, plot snapshots of the adjoint solution. ::

fig, axes = plot_snapshots(solutions, P, "uv_2d", "adjoint")
fig.savefig("burgers-time_integrated.jpg")

# .. figure:: burgers-time_integrated.jpg
#    :figwidth: 90%
#    :align: center
#
# With a time-integrated QoI, the adjoint problem
# has a source term at the right hand boundary, rather
# than a instantaneous pulse at the terminal time.
#
# This demo can also be accessed as a `Python script <burgers-time_integrated.py>`__.
