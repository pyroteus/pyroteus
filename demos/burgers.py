# Adjoint of Burgers equation
# ===========================
#
# This demo shows how to solve the adjoint of the `Firedrake`
# `Burgers equation demo <https://firedrakeproject.org/demos/burgers.py.html>`__
# using Pyroteus. There is no mesh adaptation in this demo;
# the PDE
#
# .. math::
#    \frac{\partial u}{\partial t} + (u\cdot\nabla)u - \nu\nabla^2u = 0 \quad\text{in}\quad \Omega\\
#    \nabla u\cdot n = 0 \quad\text{on}\quad \partial\Omega
#
# and its discrete adjoint are solved on a single, uniform mesh
# of the unit square, :math:`\Omega = [0, 1]^2`.
#
# We always begin by importing Pyroteus. Adjoint mode is used
# so that we have access to the discrete adjoint functionality
# due to `dolfin-adjoint`. ::

from pyroteus_adjoint import *

# Pyroteus requires a solver with four arguments: an initial
# condition, a start time, an end time and a timestep. It
# also takes two keyword arguments: a quantity of interest (QoI)
# ``qoi`` and a current value for the QoI, ``J``. The following
# pattern is used for the QoI, which will be specified later. ::

def solver(ic, t_start, t_end, dt, qoi=None, J=0):
    function_space = ic.function_space()
    dtc = Constant(dt)
    nu = Constant(0.0001)

    # Set initial condition
    u_ = Function(function_space)
    u_.assign(ic)

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
            J += qoi(u, t)
        u_.assign(u)
        t += dt
    return u_, J

# Pyroteus also requires a function for generating an initial
# condition from a function space. Note that we add the
# ``no_annotations`` decorator to initial conditions so that
# their contents aren't annotated. ::

@no_annotations
def initial_condition(function_space):
    x, y = SpatialCoordinate(function_space.mesh())
    return interpolate(as_vector([sin(pi*x), 0]), function_space)

# In line with the
# `firedrake-adjoint demo <https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/11-extract-adjoint-solutions.ipynb>`__, we choose
# a QoI which integrates the squared solution at the final time
# over the right hand boundary. ::

def end_time_qoi(sol):
    return inner(sol, sol)*ds(2)

# Now that we have the above functions defined, we move onto the
# concrete parts of the solver. As in the original demo, we use a
# :math:`\mathbb P2` function space to represent the prognostic
# variable, :math:`u`. The timestep is chosen according to the
# mesh resolution. ::

n = 32
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
end_time = 0.5
num_subintervals = 1
dt = 1/n

# The final ingredient required to solve the adjoint problem using
# Pyroteus is a ``TimePartition``. This object captures how the
# temporal domain is to be divided for the purposes of mesh
# adaptation. In our case, there is a single mesh, so the partition
# is trivial. ::

P = TimePartition(end_time, num_subintervals, dt, timesteps_per_export=2)

# Finally, we are able to call ``solve_adjoint``, which returns the
# QoI value and a dictionary of solutions for the forward and adjoint
# problems. ::

J, solutions = solve_adjoint(solver, initial_condition, end_time_qoi, V, P)

# Finally, we plot the adjoint solution at each exported timestep by
# looping over ``solutions['adjoint']``. ::

import matplotlib.pyplot as plt
rows = P.exports_per_subinterval[0] - 1
cols = P.num_subintervals
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 24//cols))
levels = np.linspace(0, 0.8, 9)
for i, adj_sols_step in enumerate(solutions['adjoint']):
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
# This demo can also be accessed as a `Python script <burgers.py>`__.
