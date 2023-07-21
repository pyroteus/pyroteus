# Burgers equation with Hessian-based mesh adaptation
# ===================================================

# Yet again, we revisit the Burgers equation example. This time, we apply Hessian-based
# mesh adaptation. The interesting thing about this demo is that, unlike the
# `previous demo <./point_discharge2d-goal_oriented.py.html>`__ and its predecessor,
# we consider the time-dependent case. Moreover, we consider a :class:`MeshSeq` with
# multiple subintervals and hence multiple meshes to adapt.
#
# As before, we copy over what is now effectively boiler plate to set up our problem. ::

from firedrake import *
from firedrake.meshadapt import adapt
from pyroteus import *
import matplotlib.pyplot as plt


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
        return {"u": F}

    return form


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space, name="u")

        # Initialise 'lagged' solution
        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})["u"]

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


n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n

num_subintervals = len(meshes)
time_partition = TimePartition(
    end_time, num_subintervals, dt, fields, timesteps_per_export=2, debug=True
)

params = MetricParameters()
mesh_seq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    parameters=params,
)

# As in the previous adaptation demos, the most important part is the adaptor function.
# The one used here takes a similar form, except that we need to handle multiple meshes
# and metrics.
#
# Since the Burgers equation has a vector solution, we recover the Hessian of each
# component at each timestep and intersect them. We then time integrate these Hessians
# to give the metric contribution from each subinterval. Given that we use a simple
# implicit Euler method for time integration in the PDE, we do the same here, too.
#
# Pyroteus provides functionality to normalise a list of metrics using *space-time*
# normalisation. This ensures that the target complexity is attained on average across
# all timesteps.
#
# Note that when adapting the mesh, we need to be careful to check whether convergence
# has already been reached on any of the subintervals. If so, the adaptation step is
# skipped. ::


def adaptor(mesh_seq, solutions):
    metrics = []
    complexities = []

    # Ramp the target average metric complexity per timestep
    base, target, iteration = 400, 1000, mesh_seq.fp_iteration
    mp = {
        "dm_plex_metric": {
            "target_complexity": ramp_complexity(base, target, iteration),
            "p": 1.0,
            "h_min": 1.0e-04,
            "h_max": 1.0,
        }
    }

    for i, mesh in enumerate(mesh_seq):
        sols = solutions["u"]["forward"][i]
        dt = mesh_seq.time_partition.timesteps[i]

        # Define the Riemannian metric
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)

        # Recover the Hessian at each timestep and time integrate
        hessians = [RiemannianMetric(P1_ten) for _ in range(2)]
        for sol in sols:
            for i, hessian in enumerate(hessians):
                hessian.set_parameters(mp)
                hessian.compute_hessian(sol[i])
                hessian.enforce_spd(restrict_sizes=True)
            metric += dt * hessians[0].intersect(hessians[1])
        metrics.append(metric)

    # Apply space time normalisation
    space_time_normalise(metrics, mesh_seq.time_partition, mp)

    # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
    for i, metric in enumerate(metrics):
        if not mesh_seq.converged[i]:
            mesh_seq[i] = adapt(mesh_seq[i], metric)
        complexities.append(metric.complexity())
    num_dofs = mesh_seq.count_vertices()
    num_elem = mesh_seq.count_elements()
    pyrint(f"fixed point iteration {iteration + 1}:")
    for i, (complexity, ndofs, nelem) in enumerate(zip(complexities, num_dofs, num_elem)):
        pyrint(
            f"  subinterval {i}, complexity: {complexity:4.0f}"
            f", dofs: {ndofs:4d}, elements: {nelem:4d}"
        )

    # Plot each intermediate adapted mesh
    fig, axes = mesh_seq.plot()
    for i, ax in enumerate(axes):
        ax.set_title(f"Mesh {i + 1}")
    fig.savefig(f"burgers-hessian_mesh{iteration + 1}.jpg")
    plt.close()

    # Since we have two subintervals, we should check if the target complexity has been
    # (approximately) reached on each subinterval
    continue_unconditionally = np.any(np.array(complexities) < 0.95 * target)
    return continue_unconditionally


# With the adaptor function defined, we can call the fixed point iteration method. ::

solutions = mesh_seq.fixed_point_iteration(adaptor)

# Here the output should look something like
#
# .. code-block:: console
#
#     fixed point iteration 1:
#       subinterval 0, complexity:  515, dofs:  714, elements: 1320
#       subinterval 1, complexity:  437, dofs:  570, elements: 1055
#     fixed point iteration 2:
#       subinterval 0, complexity:  787, dofs:  956, elements: 1786
#       subinterval 1, complexity:  644, dofs:  758, elements: 1416
#     fixed point iteration 3:
#       subinterval 0, complexity: 1044, dofs: 1225, elements: 2319
#       subinterval 1, complexity:  861, dofs:  993, elements: 1871
#     fixed point iteration 4:
#       subinterval 0, complexity: 1301, dofs: 1491, elements: 2838
#       subinterval 1, complexity: 1080, dofs: 1236, elements: 2346
#     fixed point iteration 5:
#       subinterval 0, complexity: 1300, dofs: 1538, elements: 2929
#       subinterval 1, complexity: 1081, dofs: 1259, elements: 2392
#     fixed point iteration 6:
#       subinterval 0, complexity: 1299, dofs: 1554, elements: 2960
#       subinterval 1, complexity: 1081, dofs: 1265, elements: 2403
#     fixed point iteration 7:
#       subinterval 0, complexity: 1298, dofs: 1560, elements: 2972
#       subinterval 1, complexity: 1082, dofs: 1278, elements: 2427
#     fixed point iteration 8:
#       subinterval 0, complexity: 1298, dofs: 1573, elements: 2998
#       subinterval 1, complexity: 1083, dofs: 1292, elements: 2453
#     fixed point iteration 9:
#       subinterval 0, complexity: 1297, dofs: 1575, elements: 3001
#       subinterval 1, complexity: 1084, dofs: 1300, elements: 2466
#     fixed point iteration 10:
#       subinterval 0, complexity: 1297, dofs: 1581, elements: 3013
#       subinterval 1, complexity: 1084, dofs: 1309, elements: 2484
#     fixed point iteration 11:
#       subinterval 0, complexity: 1296, dofs: 1583, elements: 3016
#       subinterval 1, complexity: 1084, dofs: 1317, elements: 2503
#     Element count converged on subinterval 0 after 11 iterations under relative tolerance 0.001.
#     fixed point iteration 12:
#       subinterval 0, complexity: 1296, dofs: 1583, elements: 3016
#       subinterval 1, complexity: 1085, dofs: 1318, elements: 2505
#     Element count converged on subinterval 1 after 12 iterations under relative tolerance 0.001.
#
# In the sample output above, we reach convergence on the first subinterval before the
# second. Notice that the DoF and element counts do not change after this point for this
# subinterval, as expected.
#
# Finally, let's plot the adapted meshes. ::

mesh_seq.plot()
axes.set_title("Adapted mesh")
fig.savefig("burgers-hessian_mesh.jpg")
plt.close()

# .. figure:: burgers-hessian_mesh.jpg
#    :figwidth: 100%
#    :align: center
#
# Recall that the Burgers problem is quasi-1D, since the analytical solution varies in
# the :math:`x`-direction, but is constant in the :math:`y`-direction. As such, we can
# affort to have lower resolution in the :math:`y`-direction in adapted meshes. That
# this occurs is clear from the plots above. The solution moves to the right, becoming
# more densely distributed near to the right-hand boundary. This can be seen by
# comparing the second mesh against the first.
#
# This demo can also be accessed as a `Python script <burgers-hessian.py>`__.
