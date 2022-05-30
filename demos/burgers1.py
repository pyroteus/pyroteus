# Adjoint of Burgers equation
# ===========================
#
# This demo solves the same problem as `the previous one
# <./burgers.py.html>`__, but making use of *dolfin-adjoint*'s
# automatic differentiation functionality in order to
# automatically form and solve discrete adjoint problems.
#
# We always begin by importing Pyroteus. Adjoint mode is used
# so that we have access to the :class:`AdjointMeshSeq` class.
# ::

from firedrake import *
from pyroteus_adjoint import *

# For ease, the field list and functions for obtaining the
# function spaces, forms, solvers, and initial conditions
# are imported from the previous demo. ::

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
#    J(u) = \int_0^1 \mathbf u(1,y,t_{\mathrm{end}})
#    \cdot \mathbf u(1,y,t_{\mathrm{end}})\;\mathrm dy,
#
# which integrates the square of the solution
# :math:`\mathbf u(x,y,t)` at the final time over the right
# hand boundary. ::


def get_qoi(mesh_seq, solutions, i):
    def end_time_qoi():
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

time_partition = TimeInterval(end_time, dt, fields, timesteps_per_export=2)

# Finally, we are able to construct an :class:`AdjointMeshSeq` and
# thereby call its :meth:`solve_adjoint` method. This computes the QoI
# value and returns a dictionary of solutions for the forward and adjoint
# problems. ::

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

# The solution dictionary is similar to :meth:`solve_forward`,
# except there are keys ``"adjoint"`` and ``"adjoint_next"``, in addition
# to ``"forward"``, ``"forward_old"``. For a given subinterval ``i`` and
# timestep index ``j``, ``solutions["adjoint"]["u"][i][j]`` contains
# the adjoint solution associated with field ``"u"`` at that timestep,
# whilst ``solutions["adjoint_next"]["u"][i][j]`` contains the adjoint
# solution from the *next* timestep (with the arrow of time going forwards,
# as usual). Adjoint equations are solved backwards in time, so this is
# effectively the "lagged" adjoint solution, in the same way that
# ``"forward_old"`` corresponds to the "lagged" forward solution.
#
# Finally, we plot the adjoint solution at each exported timestep by
# looping over ``solutions['adjoint']``. This can also be achieved using
# the plotting driver function ``plot_snapshots``.

fig, axes = plot_snapshots(
    solutions, time_partition, "u", "adjoint", levels=np.linspace(0, 0.8, 9)
)
fig.savefig("burgers1-end_time.jpg")

# .. figure:: burgers1-end_time.jpg
#    :figwidth: 50%
#    :align: center
#
# Since the arrow of time reverses for the adjoint problem, the plots
# should be read from bottom to top. The QoI acts as an impulse at the
# final time, which propagates in the opposite direction than information
# flows in the forward problem.
#
# In the `next demo <./burgers2.py.html>`__, we solve the same problem
# on two subintervals and check that the results match.
#
# This demo can also be accessed as a `Python script <burgers1.py>`__.
