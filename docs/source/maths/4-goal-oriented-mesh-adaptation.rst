=============================
Goal-oriented mesh adaptation
=============================

Error indicators
----------------

Goal-oriented mesh adaptation is the term used to refer to
mesh adaptation methods that are driven by goal-oriented
error estimates. Recall from `the error estimation section
<2-goal-oriented-error-estimation.html>`__ of the manual
that the fundamental result is the dual weighted residual
(DWR) :eq:`dwr1`, which can be written in approximate form

.. math::
    :label: dwr

    J(u)-J(u_h)\approx\rho(u_h,(u_h^*)^+),

where :math:`(u_h^*)^+` denotes an approximation of the
adjoint solution from a (finite-dimensional) superspace of
the base finite element space, i.e. :math:`V_h^+\supset V_h`.

Mesh adaptation is all about wisely using varying resolution
to equidistribute errors. This means increasing resolution
where errors are deemed to be high and reducing it where
errors are deemed to be low. We cannot immediately deduce
spatially varying information from :eq:`dwr` as it is currently
written. A simple, but effective way of extracting such
information is to compute the element-wise contributions,

.. math::
    :label: dwr_sum

    \rho(u_h,(u_h^*)^+)=\sum_{K\in\mathcal{H}}i_K,

where

.. math::
    :label: dwr_indicator

    i_K:=\rho(u_h,(u_h^*)^+)|_K.

is the so-called *error indicator* for element :math:`K`.
The second output of :meth:`~.GoalOrientedMeshSeq.indicate_errors`
contains error indicator fields -- element-wise constant fields,
whose value on :math:`K` is the indicator :math:`i_K`.

Note that so far we have only discussed how to estimate and
indicate errors for steady-state problems. The extension to
the time-dependent case is similar, in the sense that we
take a timestep-by-timestep approach in time like how we take an
element-by-element approach in space. Avoiding some minor details,
we can apply all of the subsequently discussed methodology to the
weak residual associated with a single timestep. For simple
time integration schemes like Implicit Euler, the main difference
will be that the weak residual also includes a term that
discretises the time derivative.

Adapting based on error indicators
----------------------------------

Once error indicator fields have been computed, there are many
different ways to perform mesh adaptation. One common approach is
to use a hierarchical type approach, where the mesh resolution is
locally increased in elements where the indicator value falls
below a pre-specified tolerance and is locally decreased if the
indicator values are already lower than required. This is sometimes
called "adaptive mesh refinement (AMR)", although the literature is
not entirely consistent on this. The terms "quad-tree" and "oct-tree"
are used when it is applied to quadrilateral and hexahedral meshes,
respectively. Sadly, this form of mesh adaptation is not supported
in Firedrake.

As described in the `previous section <3-metric-based.html>`__,
metric-based mesh adaptation is another approach which is currently
being integrated into Firedrake and is supported on developer branches.
In that case, we need to construct a Riemannian metric from an
error indicator field. Goalie provides a number of different
driver functions for doing this. The simplest is
:func:`~.isotropic_metric`, which takes an error indicator field
and constructs a metric that specifies how a mesh should be adapted
purely in terms of element size (not orientation or shape).
Alternative anisotropic formulations, which combine error indicator
information with curvature information from the Hessians of solution
fields are provided by :func:`~.anisotropic_metric`. See the API
documentation (and references therein) for details.

The metric-based framework has been the main backend used for
goal-oriented mesh adaptation during the development of Goalie.
However, this does not mean other approaches would not work.
Mesh movement (or :math:`r`-adaptation) has been implemented in
Firedrake on a number of occasions (such as :cite:`MCB:18,CWK+:22`)
and could plausibly be driven by goal-oriented error estimates.

Fixed point iteration loops
---------------------------

In some mesh adaptation approaches, it is common practice to adapt
the mesh multiple times until convergence is attained, in some
sense. This is often the case under metric based mesh adaptation,
for example. Goalie includes two methods to facilitate such
iterative adaptation approaches, as described in the following.

In the non-goal-oriented case, there is the
:meth:`~.MeshSeq.fixed_point_iteration` method, which accepts a
Python function that describes how to adapt the mesh as an argument.
This provides the flexibility to use different adaptation routines.
The function should take as input the :class:`~.MeshSeq` instance
and the dictionary of solution fields from each timestep. For
example, it could compute the Hessian of the solution field at each
timestep using :meth:`~.RiemannianMetric.compute_hessian`, ensure that
it is a metric using :meth:`~.RiemannianMetric.enforce_spd` and then
combine this information using averaging or intersection.
In each iteration, the PDE will be
solved over the sequence of meshes (with data transferred inbetween)
using :meth:`~.MeshSeq.solve_forward` and a Hessian-metric will be
constructed on each mesh. All of the meshes are then adapted.

The iteration is terminated according to :class:`~.AdaptParameters`
when either the pre-specified maximum iteration count
:attr:`~.AdaptParameters.maxiter` is reached, or the meshes no
longer change substantially. This is the case when none of the
corresponding element counts changes more than the relative
tolerance :attr:`~.AdaptParameters.element_rtol`.

.. note::
    The convergence criteria are not actually checked until the
    minimum iteration count :attr:`~.AdaptParameters.miniter`
    has been met.

In the goal-oriented case, the
:meth:`~.GoalOrientedMeshSeq.fixed_point_iteration` method takes a
similar form, except that
:meth:`~.GoalOrientedMeshSeq.indicate_errors` is called, rather than
:meth:`~.MeshSeq.solve_forward`. That is, the forward problem is
solved over all meshes in the sequence, then the adjoint problem is
solved over all meshes in reverse and finally goal-oriented error
indicators are computed. As such, the adaptor function depends on
these error indicators, as well as the :class:`~.MeshSeq` and
solution field dictionary (which contains solutions of both the
forward and adjoint problems).

Like with the simpler case, the goal-oriented fixed point iteration
loop is terminated if the maximum iteration count or relative
element count convergence conditions are met. However, there are two
additional convergence criteria defined in :class:`~.GoalOrientedParameters`.
Convergence is deduced if the QoI value changes between iterations
by less than :attr:`~.GoalOrientedParameters.qoi_rtol`. It is
similarly deduced if the error estimate value changes by less than
:attr:`~.GoalOrientedParameters.estimator_rtol`.


References
----------

.. bibliography:: 4-references.bib
