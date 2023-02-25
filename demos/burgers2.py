# Adjoint Burgers equation on two meshes
# =========================================
#
# This demo solves the same adjoint problem as `the previous one
# <./burgers1.py.html>`__, but now using two subintervals. There
# is still no error estimation or mesh adaptation; the same mesh
# is used in each case to verify that the framework works.
#
# Again, begin by importing Pyroteus with adjoint mode
# activated. ::

from firedrake import *
from pyroteus_adjoint import *

# Redefine the ``fields`` variable from the previous demo, as well as all the getter
# functions. ::

fields = ["u"]


def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(mesh_seq):
    def form(index, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition
        dt = Constant(P.timesteps[index])

        # Specify viscosity coefficient
        nu = Constant(0.0001)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return F

    return form


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space)

        # Initialise 'lagged' solution
        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="u")
            u_.assign(u)
            t += dt
        return {"u": u}

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {"u": interpolate(as_vector([sin(pi * x), 0]), fs)}


def get_qoi(mesh_seq, solutions, i):
    def end_time_qoi():
        u = solutions["u"]
        return inner(u, u) * ds(2)

    return end_time_qoi


# The solver, initial condition and QoI may be imported from the
# previous demo. The same basic setup is used. The only difference
# is that the :class:`MeshSeq` contains two meshes. ::

n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n

# This time, the ``TimePartition`` is defined on **two** subintervals. ::

num_subintervals = len(meshes)
time_partition = TimePartition(
    end_time, num_subintervals, dt, fields, timesteps_per_export=2, debug=True
)
mesh_seq = AdjointMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

# Recall that :func:`solve_forward` runs the solver on each subinterval and
# uses conservative projection to transfer inbetween. This also happens in
# the forward pass of :func:`solve_adjoint`, but is followed by running the
# *adjoint* of the solver on each subinterval *in reverse*. The adjoint of
# the conservative projection operator is applied to transfer adjoint solution
# data between meshes in this case. If you think about the matrix
# representation of a projection operator then this effectively means taking
# the transpose. Again, the meshes (and hence function spaces) are identical,
# so the transfer is just the identity.
#
# Snapshots of the adjoint solution are again plotted using the
# :func:`plot_snapshots` utility function. ::

fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "u", "adjoint", levels=np.linspace(0, 0.8, 9)
)
fig.savefig("burgers2-end_time.jpg")

# .. figure:: burgers2-end_time.jpg
#    :figwidth: 90%
#    :align: center
#
# The adjoint solution fields at each time level appear to match
# those due to the previous demo at each timestep. That they actually
# do coincide is checked in Pyroteus' test suite.
#
# .. rubric:: Exercise
#
# Note that the keyword argument ``diagonal="left"`` was passed to the
# ``UnitSquareMesh`` constructor in this example, defining which way
# the diagonal lines in the uniform mesh should go. Instead of having
# both function spaces defined on this mesh, try defining the second
# one in a :math:`\mathbb P2` space defined on a **different** mesh
# which is constructed with ``diagonal="right"``. How does the adjoint
# solution change when the solution is trasferred between different
# meshes? In this case, the mesh-to-mesh transfer operations will no
# longer simply be identities.
#
# In the `next demo <./burgers_time_integrated.py.html>`__, we solve
# the same problem but with a QoI involving an integral in time, as
# well as space.
#
# This demo can also be accessed as a `Python script <burgers2.py>`__.
