# Adjoint of Burgers equation on two meshes
# =========================================
#
# This demo solves the same problem as the previous two,
# but now using two subintervals. There is still no mesh
# adaptation; the same mesh is used in each case to verify
# that the framework works.
#
# Again, begin by importing Pyroteus with adjoint mode
# activated. ::

from firedrake import *
from pyroteus_adjoint import *

# The solver, initial condition and QoI may be imported from the
# previous demo. The same basic setup is used. The only difference
# is that there are **two** function spaces associated with the
# velocity space (which actually coincide). They are represented
# in a list. ::

from burgers1 import (
    fields,
    get_function_spaces,
    get_form,
    get_solver,
    get_initial_condition,
    get_qoi,
)

n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n

# This time, the ``TimePartition`` is defined on **two** subintervals,
# associated with the two function spaces. ::

num_subintervals = 2
P = TimePartition(
    end_time, num_subintervals, dt, fields, timesteps_per_export=2, debug=True
)
mesh_seq = AdjointMeshSeq(
    P,
    meshes,
    get_function_spaces,
    get_initial_condition,
    get_form,
    get_solver,
    get_qoi,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

# Finally, plot snapshots of the adjoint solution. ::

fig, axes = plot_snapshots(solutions, P, "u", "adjoint", levels=np.linspace(0, 0.8, 9))
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
# Note that the keyword argument ``diagonal='left'`` was passed to the
# ``UnitSquareMesh`` constructor in this example, defining which way
# the diagonal lines in the uniform mesh should go. Instead of having
# both function spaces defined on this mesh, try defining the second
# one in a :math:`\mathbb P2` space defined on a **different** mesh
# which is constructed with ``diagonal='right'``. How does the adjoint
# solution change when the solution is trasferred between different
# meshes?
#
# In the `next demo <./burgers-time_integrated.py.html>`__, we solve
# the same problem but with a QoI involving an integral in time, as
# well as space.
#
# This demo can also be accessed as a `Python script <burgers2.py>`__.
