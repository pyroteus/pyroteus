==============================
Goal-oriented error estimation
==============================

In this manual page, it is assumed that you have read and understood
the `dolfin-adjoint
<http://www.dolfin-adjoint.org/en/latest/documentation/maths/index.html>`__
training material on adjoint methods.

Pyroteus has been designed with time-dependent problems in mind. However,
the exposition of goal-oriented error estimation is most clearly presented
in the steady-state case. Therefore, suppose we have a "forward" PDE that
contains only derivatives in space and is written in the residual form,

.. math::
    :label: forward

    F(u) = 0,\quad u\in V,

where :math:`u` is the solution, which lives in a function space :math:`V`.
In addition, suppose that there exists a diagnostic quantity of interest
(QoI),

.. math::
    :label: qoi

    J:V\rightarrow\mathbb R,

for which we would like to accurately evaluate :math:`J(u)`. The adjoint
problem associated with :math:`J` is then given by

.. math::
    :label: adjoint

    \frac{\partial F}{\partial u}^Tu^*=\frac{\partial J}{\partial u}^T,
    \quad u^*\in V,

where :math:`u^*` is the *adjoint solution*, which also lives in :math:`V`.

Consider now a finite-dimensional subspace :math:`V_h\subset V` and suppose
that we have a weak formulation of the forward problem given by

.. math::
    :label: weak_forward

    \rho(u_h,v)=0,\quad\forall v\in V_h,

where :math:`u_h` is the approximate (weak) forward solution. Here
:math:`\rho(u_h,\cdot)` is often called the "weak residual" of the forward
problem. This is the equation that we solve when we call
:meth:`~.MeshSeq.solve_forward`. Similarly, suppose we have a weak formulation
of the adjoint problem with approximate (weak) adjoint solution :math:`u_h^*\in V_h`:

.. math::
    :label: weak_adjoint

    \rho^*(u_h^*,v)=0,\quad\forall v\in V_h,

where :math:`\rho^*(u_h^*,\cdot)` is the weak residual of the adjoint problem.
This is the equation that we solve when we call
:meth:`~.AdjointMeshSeq.solve_adjoint`.

Recall that we seek to accurately evaluate :math:`J` using a finite element
method. This is the same as saying that we seek to minimise the "QoI error"
:math:`J(u)-J(u_h)`. The fundamental result in goal-oriented error estimation
is the first order *dual weighted residual* result :cite:`BR:01`,

.. math::
    :label: dwr1

    J(u)-J(u_h)=\rho(u_h,u^*)+R^{(2)},

which relates the QoI error to the forward weak residual with the test function
replaced by the adjoint solution. This result is "first order" in the sense
that the remainder term :math:`R^{(2)}` is quadratic in the forward and adjoint
discretisation errors :math:`u-u_h` and :math:`u^*-u_h^*`. There also exists
a second order result,

.. math::
    :label: dwr2

    J(u)-J(u_h)=\frac12\rho(u_h,u^*)+\frac12\rho^*(u_h^*,u)+R^{(3)},

with remainder term :math:`R^{(3)}` that is cubic in the forward and adjoint
discretisation errors. We refer to the part of the RHS of each equation without
the remainder term a *dual weighted residual error estimate*, since it approximates
the QoI error.

Note that the first order DWR result :eq:`dwr1` replaces the test function with
the *true* adjoint solution, :math:`u^*`. Further, the second order result
:eq:`dwr2` also includes the *true* forward solution. Neither of these quantities
are known in practice. Therefore, we can usually only evaluate dual weighted
residual error estimates in an approximate sense. Typically, this means approximating
the true adjoint and/or forward solution using a higher order method. A simple -- but
computationally intensive -- approach is to solve the appropriate equation again in a
globally "enriched" finite element space. For example, on a uniformly refined mesh or
in a function space with higher polynomial order. This can be achieved in Pyroteus
using :meth:`~.GoalOrientedMeshSeq.indicate_errors`. See `the Burgers error estimation demo
<../demos/burgers_ee.py.html>`__ for example usage.

References
----------

.. bibliography:: 2-references.bib
