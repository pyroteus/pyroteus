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

from firedrake import *
from pyroteus import *

# In this problem, we have a single prognostic variable,
# :math:`\mathbf u`. Its name is recorded in a list of
# fields. ::

fields = ["u"]

# For each such field, we need to be able to specify how to
# build a :class:`FunctionSpace`, given some mesh. Since there
# could be more than one field, function spaces are given as a
# dictionary, indexed by the prognostic solution field names. ::


def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


# In order to solve PDEs using the finite element method, we
# require a weak form. For this, Pyroteus requires a function
# that maps the :class:`MeshSeq` index and a dictionary of
# solution data to the form. For each field, the dictionary
# should provide a tuple containing the solution :class:`Function`\s
# from the current timestep and the previous one. The form should be
# returned inside its own dictionary, with an entry for each equation
# being solved for.
#
# Timestepping information associated with a given subinterval
# can be accessed via the :attr:`TimePartition` attribute of
# the :class:`MeshSeq`. ::


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


# We have a weak form. In order to solve the PDE, we need to choose
# a time integration routine and solver parameters for the underlying
# linear and nonlinear systems. This is achieved using a function
# whose inputs are the :class:`MeshSeq` index and a dictionary of
# initialisation data. For the :math:`0^{th}` index, this will be
# provided by the initial conditions, otherwise it will be transferred
# from the previous mesh in the sequence.
#
# Note that it is important that the PDE solve is labelled
# with an ``ad_block_tag`` which matches the corresponding
# prognostic variable name. It is also important that the
# lagged solution field be given a particular name: the field name,
# appended by ``'_old'``. ::


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space)

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


# The dictionary usage may seem cumbersome when applied to such a
# simple problem, but it comes in handy when solving adjoint
# problems associated with coupled systems of equations.
#
# Pyroteus also requires a function for generating an initial
# condition from the function space defined on the
# :math:`0^{th}` mesh. ::


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {"u": interpolate(as_vector([sin(pi * x), 0]), fs)}


# Now that we have the above functions defined, we move onto the
# concrete parts of the solver. To begin with, we require a
# sequence of meshes, simulation end time and a timestep. ::

n = 32
meshes = [
    UnitSquareMesh(n, n),
    UnitSquareMesh(n, n),
]
end_time = 0.5
dt = 1 / n

# These can be used to create a :class:`TimePartition` for the
# problem with two subintervals. ::

num_subintervals = len(meshes)
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    timesteps_per_export=2,
)

# Finally, we are able to construct a :class:`MeshSeq` and
# solve Burgers equation over the meshes in sequence. ::

mesh_seq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
)
solutions = mesh_seq.solve_forward()

# During the :func:`solve_forward` call, the solver that was provided
# is applied on the first subinterval. The forward solution at the end
# of that subinterval is transferred to the mesh associated with the
# second subinterval and used as an initial condition for the same solver
# applied again there. Pyroteus uses a conservative interpolation
# operator to transfer solution data between the two meshes. In this
# example, the meshes (and hence function spaces) are identical so the
# projection operation will in fact be the identity.
#
# The output is a nested dictionary of solution data, indexed by
# solution type (``"forward"`` or ``"forward_old"``) and then field name
# (here ``"u"``). The contents of the inner dictionaries are lists
# containing lists of solution :class:`Function`\s, indexed first by
# subinterval and then by timestep. That is,
# ``solutions["forward"]["u"][i][j]`` contains the forward solution
# associated with field ``"u"`` at the ``j``-th timestep of
# subinterval ``i``. Similarly, ``solutions["forward_old"]["u"][i][j]``
# contains the forward solution from the timestep prior.
#
# For the purposes of this demo, we plot the solution at each exported
# timestep using the plotting driver function :func:`plot_snapshots`. ::

fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "u", "forward", levels=np.linspace(0, 1, 9)
)
fig.savefig("burgers.jpg")

# .. figure:: burgers.jpg
#    :figwidth: 90%
#    :align: center
#
# We see that the initial sinusoid is nonlinearly advected Eastwards.
#
# In the `next demo <./burgers1.py.html>`__, we use Pyroteus to
# automatically solve the adjoint problem associated with Burgers
# equation.
#
# This demo can also be accessed as a `Python script <burgers.py>`__.
