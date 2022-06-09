# Partitioning a time interval
# ============================
#
# Pyroteus is focused on solving PDEs and
# their adjoints over some time interval
# on sequences of meshes.
#
# Suppose we have a time interval
#
# .. math::
#    \mathcal T := (t_{\mathrm{start}}, t_{\mathrm{end}}],
#
# for simulation start and end times
# :math:`0 \leq t_{\mathrm{start}} < t_{\mathrm{end}}`.
# One of the fundamental objects in
# Pyroteus is the :class:`TimePartition`,
# which subdivides :math:`\mathcal T` into
# :math:`n\in\mathbb N` subintervals,
#
# .. math::
#    \mathcal T = \cap_{k=1}^n (t^{(k-1)}, t^{(k)}],
#
# where :math:`t^{(0)}=t_{\mathrm{start}}`
# and :math:`t^{(n)}=t_{\mathrm{end}}`. In
# a time-dependent mesh adaptive simulation,
# each subinterval will be associated with a
# single mesh.
#
# We begin by importing Pyroteus. ::

from pyroteus import *

# To create a :class:`TimePartition`, we
# need at least four ingredients:
#
# * the end time;
# * the number of subintervals;
# * the timestep on each subinterval;
# * a list of field names for the solution components.
#
# If the start time is not set then it is
# assumed to be zero.
#
# The simplest possible partition is to
# consider a single subinterval. Suppose
# the interval is :math:`(0,1]`, the timestep
# is :math:`\Delta t=\frac18` and the field
# name is `"solution"`. This is written in
# code as ::

end_time = 1.0
num_subintervals = 1
dt = 0.125
fields = ["solution"]

# With these definitions, we should get
# one subinterval of :math:`(0,1]` containing
# eight timesteps. When constructing a
# :class:`TimePartition` (or any other object),
# it is often useful to use Pyroteus' debugging
# mode. This is specified using :func:`set_log_level`. ::

set_log_level(DEBUG)
P = TimePartition(end_time, num_subintervals, dt, fields)

# Notice that one of the things which is printed
# out is ``timesteps_per_export``, which controls
# how frequently data is to be exported to file
# during a simulation. It defaults to one, but may
# be specified as a keyword argument.
#
# Based on the above values, the
# :class:`TimePartition` computes the number
# of exports per subinterval. Note that this
# number assumes that exports occur at both
# the start and end of the subinterval, so the
# value may be one larger than you expect.
#
# This partition isn't particularly interesting.
# Let's try constructing a new one with more
# than one subinterval. ::

num_subintervals = 2
P = TimePartition(end_time, num_subintervals, dt, fields, timesteps_per_export=2)

# In some problems, the dynamics evolve such
# that different timesteps are suitable during
# different parts of the simulation. To account
# for that, it is possible to specify a list of
# timesteps corresponding to each subinterval. ::

dt = [0.125, 0.0625]
P = TimePartition(end_time, num_subintervals, dt, fields, timesteps_per_export=2)

# Note that this means that there are more
# exports in the second subinterval than the first.
# This can be remedied by also setting
# ``timesteps_per_export`` as a list. ::

P = TimePartition(end_time, num_subintervals, dt, fields, timesteps_per_export=[2, 4])

# So far, we have assumed that the subintervals
# are of uniform length. This need not be the case.
# To set up a :class:`TimePartition` with non-uniform
# subintervals, the subintervals need to be passed
# to the constructor as a list of tuples. ::

subintervals = [(0.0, 0.75), (0.75, 1.0)]
P = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    timesteps_per_export=[2, 4],
    subintervals=subintervals,
)

# In the `next demo <./mesh_seq.py.html>`__, we
# learn how to build a :class:`MeshSeq` object
# on top of a partitioned time interval, so that
# we can solve PDEs on multiple meshes.
#
# This demo can also be accessed as a `Python script <time_partition.py>`__.
