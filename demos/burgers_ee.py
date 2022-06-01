# Error estimation for Burgers equation
# =====================================

# So far, we have learnt how to set up :class:`MeshSeq`\s and solve
# forward and adjoint problems. In this demo, we use this functionality
# to perform goal-oriented error estimation.
#
# The fundamental result in goal-oriented error estimation is the
# *dual-weighted residual*,
#
# .. math::
#    J(u)-J(u_h)\approx\rho(u_h,u^*),
#
# where :math:`u` is the solution of a PDE with weak residual
# :math:`\rho(\cdot,\cdot)`, :math:`u_h` is a finite element solution
# and :math:`J` is the quantity of interest (QoI). Here, the *exact*
# solution :math:`u^*` of the associated adjoint problem replaces the test
# function in the second argument of the weak residual. In practice,
# we do not know what this is, of course. As such, it is common practice
# to evaluate the dual weighted residual by approximating the true adjoint
# solution in an enriched finite element space. That is, a superspace,
# obtained by adding more degrees of freedom to the base space. This could
# be done by solving global or local auxiliary PDEs, or by applying patch
# recovery type methods. Currently, only global enrichment is supported in
# Pyroteus.
#
# First, import the various functions and create a sequence of meshes
# and a :class:`TimePartition`. ::

from firedrake import *
from pyroteus_adjoint import *
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
num_subintervals = len(meshes)
time_partition = TimePartition(
    end_time, num_subintervals, dt, fields, timesteps_per_export=2, debug=True
)

# A key difference between this demo and the previous ones is that we need to
# use a :class:`GoalOrientedMeshSeq` to access the goal-oriented error estimation
# functionality. Note that :class:`GoalOrientedMeshSeq` is a subclass of
# :class:`AdjointMeshSeq`, which is a subclass of :class:`MeshSeq`. ::

mesh_seq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)

# Given the description of the PDE problem in the form of a :class:`GoalOrientedMeshSeq`,
# Pyroteus is able to extract all of the relevant information to automatically compute
# error estimators. During the computation, we solve the forward and adjoint equations
# over the mesh sequence, as before. In addition, we solve the adjoint problem again
# in an *enriched* finite element space. Currently, Pyroteus supports uniform refinement
# of the meshes (:math:`h`-refinement) or globally increasing the polynomial order
# (:math:`p`-refinement). Choosing one (or both) of these as the ``"enrichment_method"``,
# we are able to compute error indicator fields as follows. ::

solutions, indicators = mesh_seq.indicate_errors(
    enrichment_kwargs={"enrichment_method": "h"}
)

# An error indicator field :math:`i` takes constant values on each mesh element, say
# :math:`i_K` for element :math:`K` of mesh :math:`\mathcal H`. It decomposes
# the global error estimator :math:`\epsilon` into its local contributions.
#
# .. math::
#    \epsilon = \sum_{K\in\mathcal H}i_K \approx \rho(u_h,u^*).
#
# For the purposes of this demo, we plot the solution at each exported
# timestep using the plotting driver function :func:`plot_indicator_snapshots`. ::

fig, axes = plot_indicator_snapshots(indicators, time_partition, levels=50)
fig.savefig("burgers-ee.jpg")

# .. figure:: burgers-ee.jpg
#    :figwidth: 90%
#    :align: center
#
# We observe that the contributions to the QoI error are estimated to be much higher in
# the right-hand part of the domain than the left. This makes sense, becase the QoI is
# evaluated along the right-hand boundary and we have already seen that the magnitude
# of the adjoint solution tends to be larger in that region, too.
#
# .. rubric:: Exercise
#
# Try running the demo again, but with a ``"time_integrated"`` QoI, rather than an
# ``"end_time"`` one. How do the error indicator fields change in this case?
#
# This demo can also be accessed as a `Python script <burgers_ee.py>`__.
