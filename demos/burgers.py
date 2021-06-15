# Burgers equation on a sequence of meshes
# ========================================

# This demo shows how to solve the `Firedrake`
# `Burgers equation demo <https://firedrakeproject.org/demos/burgers.py.html>`__
# on a sequence of meshes using Pyroteus.
# The PDE
#
# .. math::
#
#    \frac{\partial\mathbf u}{\partial t}
#    + (\mathbf u\cdot\nabla)\mathbf u
#    - \nu\nabla^2\mathbf u = \boldsymbol0 \quad\text{in}\quad \Omega\\
#
#    (\widehat{\mathbf n}\cdot\nabla)\mathbf u
#    = \boldsymbol0 \quad\text{on}\quad \partial\Omega
#
# is solved on two meshes of the unit square,
# :math:`\Omega = [0, 1]^2`. The forward solution is initialised
# as a sine wave and is nonlinearly advected to the right hand
# side. See the Firedrake demo for details on the discretisation used.
#
# We always begin by importing Pyroteus. ::

from pyroteus import *

# In this problem, we have a single prognostic variable,
# :math:`\mathbf u`. Its name is recorded in a list of
# fields. ::

fields = ['uv_2d']

# First, we specify how to build a :class:`FunctionSpace`,
# given some mesh. Function spaces are given as a dictionary,
# labelled by the prognostic solution field names. The
# function spaces will be built upon the meshes contained
# inside an :class:`MeshSeq` object, which facilitates
# solving PDEs on a sequence of meshes.


def get_function_spaces(mesh):
    return {'uv_2d': VectorFunctionSpace(mesh, "CG", 2)}


# Pyroteus requires a solver with two arguments: the index
# within the :class:`MeshSeq` and a dictionary containing a
# starting value :class:`Function` for each prognostic solution.
#
# Timestepping information associated with the mesh within
# the sequence can be accessed via the :attr:`TimePartition`
# attribute of the :class:`MeshSeq`.
#
# The following pattern is used for the QoI, which will be
# specified later. The dictionary usage may seem cumbersome when
# applied to such a simple problem, but it comes in handy when
# solving adjoint problems associated with coupled systems of
# equations. It is important that the PDE solve is labelled
# with an ``options_prefix`` which matches the corresponding
# prognostic variable name. ::


def get_solver(mesh_seq):

    def solver(index, ic):
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        function_space = mesh_seq.function_spaces['uv_2d'][index]

        # Specify constants
        dtc = Constant(dt)
        nu = Constant(0.0001)

        # Set initial condition
        u_ = Function(function_space, name='uv_2d_old')
        u_.assign(ic['uv_2d'])

        # Setup variational problem
        v = TestFunction(function_space)
        u = Function(function_space)
        F = inner((u - u_)/dtc, v)*dx \
            + inner(dot(u, nabla_grad(u)), v)*dx \
            + nu*inner(grad(u), grad(v))*dx

        # Time integrate from t_start to t_end
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, options_prefix='uv_2d')
            u_.assign(u)
            t += dt
        return {'uv_2d': u}

    return solver


# Pyroteus also requires a function for generating an initial
# condition from a function space. ::


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces['uv_2d'][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {'uv_2d': interpolate(as_vector([sin(pi*x), 0]), fs)}


# Now that we have the above functions defined, we move onto the
# concrete parts of the solver, which mimic the original demo. ::

n = 32
meshes = [
    UnitSquareMesh(n, n),
    UnitSquareMesh(n, n),
]
end_time = 0.5
dt = 1/n

# Create a :class:`TimePartition` for the problem with two
# subintervals. ::

num_subintervals = 2
P = TimePartition(
    end_time, num_subintervals, dt, fields, timesteps_per_export=2,
)

# Finally, we are able to construct a :class:`MeshSeq` and
# solve Burgers equation over the meshes in sequence. ::

mesh_seq = MeshSeq(
    P, meshes, get_function_spaces, get_initial_condition, get_solver
)
solutions = mesh_seq.solve_forward()

# Finally, we plot the solution at each exported timestep by
# looping over ``solutions['forward']``. This can be achieved using
# the plotting driver function ``plot_snapshots``.

fig, axes = plot_snapshots(solutions, P, 'uv_2d', 'forward', levels=np.linspace(0, 1, 9))
fig.savefig("burgers.jpg")

# .. figure:: burgers.jpg
#    :figwidth: 90%
#    :align: center
#
# An initial sinusoid is nonlinearly advected Eastwards.
#
# In the `next demo <./burgers1.py.html>`__, we use Pyroteus to
# automatically solve the adjoint problem associated with Burgers
# equation.
#
# This demo can also be accessed as a `Python script <burgers.py>`__.
