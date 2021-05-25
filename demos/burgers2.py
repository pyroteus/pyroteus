# Adjoint of Burgers equation on two meshes
# =========================================
#
# This demo solves the same problem as `the previous one
# <./burgers.py.html>`__, but now using two subintervals. There is
# still no mesh adaptation; the same mesh is used in each
# case to verify that the framework works.
#
# Again, begin by importing Pyroteus. ::

from pyroteus_adjoint import *

# The solver, initial condition and QoI may be imported from the
# previous demo. The same basic setup is used. ::

from burgers import solver, initial_condition, end_time_qoi

n = 32
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
end_time = 0.5
dt = 1/n

# This time, the ``TimePartition`` is defined on *two* subintervals
# and with a list of *two* function spaces (which actually coincide).
# The ``debug=True`` keyword argument is useful for checking that
# the partition has been created as desired.::

num_subintervals = 2
P = TimePartition(end_time, num_subintervals, dt, timesteps_per_export=2, debug=True)
J, solutions = solve_adjoint(solver, initial_condition, end_time_qoi, [V, V], P)

# Solution plotting is much the same, but with some minor tweaks to
# get the two subintervals side by side. ::

import matplotlib.pyplot as plt
rows = P.exports_per_subinterval[0] - 1
cols = P.num_subintervals
fig, axes = plt.subplots(rows, cols, sharex='col', figsize=(6*cols, 24//cols))
levels = np.linspace(0, 0.8, 9)
for i, adj_sols_step in enumerate(solutions['adjoint']):
    ax = axes[0, i]
    ax.set_title(f"Mesh {i+1}")
    for j, adj_sol in enumerate(adj_sols_step):
        ax = axes[j, i]
        tricontourf(adj_sol, axes=ax, levels=levels)
        ax.annotate(
            f"t={i*end_time/cols + j*P.timesteps_per_export[i]*P.timesteps[i]:.2f}",
            (0.05, 0.05), color='white',
        )
plt.tight_layout()
plt.savefig("burgers2-end_time.jpg")

# .. figure:: burgers2-end_time.jpg
#    :figwidth: 90%
#    :align: center
#
# The adjoint solution fields at each time level appear to match
# those due to the previous demo at each timestep. That they actually
# do coincide is checked in Pyroteus' test suite.
#
# This demo can also be accessed as a `Python script <burgers2.py>`__.
