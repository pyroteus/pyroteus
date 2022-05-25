# Adjoint of Burgers equation
# ===========================
#
# This demo solves the same problem as `the previous one
# <./burgers.py.html>`__, but also making use of
# dolfin-adjoint's automatic differentiation functionality
# in order to automatically form and solve discrete adjoint
# problems.
#
# We always begin by importing Pyroteus. Adjoint mode is used
# so that we have access to the discrete adjoint functionality
# due to `dolfin-adjoint`. ::

from firedrake import *
from pyroteus_adjoint import *

# The solver, initial condition and function spaces may be
# imported from the previous demo. ::

from burgers import (
    fields,
    get_function_spaces,
    get_form,
    get_solver,
    get_initial_condition,
)

# In line with the
# `firedrake-adjoint demo
# <https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/11-extract-adjoint-solutions.ipynb>`__,
# we choose the QoI
#
# .. math::
#    J(u) = \int_0^1 \mathbf u(1,y,T_{\mathrm{end}})
#    \cdot \mathbf u(1,y,T_{\mathrm{end}})\;\mathrm dy,
#
# which integrates the square of the solution
# :math:`\mathbf u(x,y,t)` at the final time over the right
# hand boundary. ::


def get_qoi(mesh_seq, i):
    def end_time_qoi(solutions):
        u = solutions["u"]
        return inner(u, u) * ds(2)

    return end_time_qoi


# Now that we have the above functions defined, we move onto the
# concrete parts of the solver, which mimic the original demo. ::

n = 32
mesh = UnitSquareMesh(n, n)
end_time = 0.5
dt = 1 / n

# Another requirement to solve the adjoint problem using
# Pyroteus is a :class:`TimePartition`. In our case, there is a
# single mesh, so the partition is trivial and we can use the
# :class:`TimeInterval` constructor. ::

P = TimeInterval(end_time, dt, fields, timesteps_per_export=2)

# Finally, we are able to construct an :class:`AdjointMeshSeq` and
# thereby call its :attr:`solve_adjoint` method. This computes the QoI
# value and returns a dictionary of solutions for the forward and adjoint
# problems. The solution dictionary has keys ``'forward'``, ``'forward_old'``,
# ``'adjoint'`` and ``'adjoint_next'`` and arrays as values. When passed
# an index corresponding to a particular exported timestep, the array
# entries correspond to the current forward solution, the forward
# solution at the previous timestep, the current adjoint solution and
# the adjoint solution at the next timestep, respectively. ::

mesh_seq = AdjointMeshSeq(
    P,
    mesh,
    get_function_spaces,
    get_initial_condition,
    get_form,
    get_solver,
    get_qoi,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

# Finally, we plot the adjoint solution at each exported timestep by
# looping over ``solutions['adjoint']``. This can also be achieved using
# the plotting driver function ``plot_snapshots``.

fig, axes = plot_snapshots(solutions, P, "u", "adjoint", levels=np.linspace(0, 0.8, 9))
fig.savefig("burgers1-end_time.jpg")

# .. figure:: burgers1-end_time.jpg
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
# This demo can also be accessed as a `Python script <burgers1.py>`__.
