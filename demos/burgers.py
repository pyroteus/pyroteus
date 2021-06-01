# Adjoint of Burgers equation
# ===========================
#
# This demo shows how to solve the adjoint of the `Firedrake`
# `Burgers equation demo <https://firedrakeproject.org/demos/burgers.py.html>`__
# using Pyroteus. There is no mesh adaptation in this demo;
# the PDE
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
# and its discrete adjoint are solved on a single, uniform mesh
# of the unit square, :math:`\Omega = [0, 1]^2`. The forward
# solution is initialised as a sine wave and is nonlinearly
# advected to the right hand side. See the Firedrake demo for
# details on the discretisation used.
#
# We always begin by importing Pyroteus. Adjoint mode is used
# so that we have access to the discrete adjoint functionality
# due to `dolfin-adjoint`. ::

from pyroteus_adjoint import *

# In this problem, we have a single prognostic variable,
# :math:`\mathbf u`. Its name is recorded in a list of
# fields. ::

fields = ['uv_2d']

# Pyroteus requires a solver with four arguments: a dictionary
# containing initial conditions for each prognostic solution,
# a start time, an end time and a timestep. It also takes two
# keyword arguments: a quantity of interest (QoI) ``qoi`` and a
# current value for the QoI, ``J``. The following pattern is
# used for the QoI, which will be specified later. The
# dictionary usage may seem cumbersome when applied to such a
# simple problem, but it comes in handy when solving adjoint
# problems associated with coupled systems of equations. ::


def solver(ic, t_start, t_end, dt, qoi=None, J=0):
    function_space = ic['uv_2d'].function_space()
    dtc = Constant(dt)
    nu = Constant(0.0001)

    # Set initial condition
    u_ = Function(function_space)
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
        solve(F == 0, u)
        if qoi is not None:
            J += qoi({'uv_2d': u}, t)
        u_.assign(u)
        t += dt
    return {'uv_2d': u_}, J


# Pyroteus also requires a function for generating an initial
# condition from a function space. Note that we add the
# ``no_annotations`` decorator to initial conditions so that
# their contents aren't annotated. ::


@no_annotations
def initial_condition(function_space):
    fs = function_space['uv_2d'][0]
    x, y = SpatialCoordinate(fs.mesh())
    return {'uv_2d': interpolate(as_vector([sin(pi*x), 0]), fs)}


# In line with the
# `firedrake-adjoint demo
# <https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/11-extract-adjoint-solutions.ipynb>`__, we choose the QoI
#
# .. math::
#    J(u) = \int_0^1 \mathbf u(1,y,T_{\mathrm{end}})
#    \cdot \mathbf u(1,y,T_{\mathrm{end}})\;\mathrm dy,
#
# which integrates the square of the solution
# :math:`\mathbf u(x,y,t)` at the final time over the right
# hand boundary. ::


def end_time_qoi(sol):
    u = sol['uv_2d']
    return inner(u, u)*ds(2)


# Now that we have the above functions defined, we move onto the
# concrete parts of the solver, which mimic the original demo.
# Function spaces are given as a dictionary, indexed by the
# prognostic solution field names. ::

n = 32
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
function_spaces = {'uv_2d': V}
end_time = 0.5
dt = 1/n

# The final ingredient required to solve the adjoint problem using
# Pyroteus is a ``TimePartition``. This object captures how the
# temporal domain is to be divided for the purposes of mesh
# adaptation. In our case, there is a single mesh, so the partition
# is trivial. ::

num_subintervals = 1
P = TimePartition(end_time, num_subintervals, dt, fields, timesteps_per_export=2)

# Finally, we are able to call ``solve_adjoint``, which returns the
# QoI value and a dictionary of solutions for the forward and adjoint
# problems. The solution dictionary has keys `'forward'`, `'forward_old'`,
# `'adjoint'` and `'adjoint_next'` and arrays as values. When passed
# an index corresponding to a particular exported timestep, the array
# entries correspond to the current forward solution, the forward
# solution at the previous timestep, the current adjoint solution and
# the adjoint solution at the next timestep, respectively. ::

J, solutions = solve_adjoint(solver, initial_condition, end_time_qoi, function_spaces, P)

# Finally, we plot the adjoint solution at each exported timestep by
# looping over ``solutions['adjoint']``. ::

import matplotlib.pyplot as plt
rows = P.exports_per_subinterval[0] - 1
cols = P.num_subintervals
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 24//cols))
levels = np.linspace(0, 0.8, 9)
for i, adj_sols_step in enumerate(solutions.uv_2d.adjoint):
    ax = axes[0]
    ax.set_title(f"Mesh {i+1}")
    for j, adj_sol in enumerate(adj_sols_step):
        ax = axes[j]
        tricontourf(adj_sol, axes=ax, levels=levels)
        ax.annotate(
            f"t={i*end_time/cols + j*P.timesteps_per_export[i]*P.timesteps[i]:.2f}",
            (0.05, 0.05), color='white',
        )
plt.tight_layout()
plt.savefig("burgers-end_time.jpg")

# .. figure:: burgers-end_time.jpg
#    :figwidth: 50%
#    :align: center
#
# Where the arrow of time progresses forwards for Burgers equation,
# it reverses for its adjoint. As such, the plots should be read from
# bottom to top. The QoI acts as an impulse at the final time, which
# propagates in the opposite direction than information flows in the
# forward problem.
#
# In the `next demo <./burgers2.py.html>`__, we solve the same problem
# on two subintervals and check that the results match.
#
# This demo can also be accessed as a `Python script <burgers.py>`__.
