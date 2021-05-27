============================
Metric-based mesh adaptation
============================

Metric spaces
-------------

The metric-based mesh adaptation framework has its
roots in Riemannian geometry - in particular,
Riemannian metric spaces.

A `metric space`
:math:`(\mathcal V,\underline{\mathbf M})` consists
of a vector space :math:`\mathcal V` and a `metric`
:math:`\underline{\mathbf M}` defined upon it. Under
the assumption that :math:`\mathcal V=\mathbb R^n`
for some :math:`n\in\mathbb N`, :math:`\mathcal M` can
be represented as an :math:`n\times n` symmetric
positive-definite (SPD) matrix. From the metric, we
may deduce a distance function and various geometric
quantities related to the metric space.

Note that the definition above assumes that the metric
takes a fixed value. A `Riemannian metric space`,
on the other hand, is defined point-wise on a domain
:math:`Omega`, such that its value is an :math:`n\times n`
SPD matrix at each :math:`\mathbf x\in\Omega`. We
use the notation
:math:`\mathcal M=\{\underline{\mathbf M}(\mathbf x)\}_{\mathbf x\in\Omega}`
for the Riemannian metric. Throughout this documentation,
the term `metric` should be understood as referring
specifically to a Riemannian metric.


Geometric interpretation
------------------------

`[To do: ellipses and ellipsoids]`

`[To do: decompositions - refer to density_and_quotients]`


Continuous mesh analogy
-----------------------

`[To do]`

Metric complexity may be computed in Pyroteus using
the function ``metric_complexity``.


Mesh adaptation
---------------

The idea of metric-based mesh adaptation is to use
a Riemannian metric space `within` the mesher.

`[To do: unit mesh, quasi-unit mesh]`


Operations on metrics
---------------------

In order to use metrics to drive mesh adaptation
algorithms for solving real problems, they must
first be made relevant. Metrics should be
normalised in order to account for domain geometry,
dimensional scales and other properties of the
problem, such as the extent to which it is
multi-scale.

In Pyroteus, normalisation is performed in the
:math:`L^p` sense.

`[To do: space normalisation]`

In Pyroteus, normalisation in space is achieved
through the function ``space_normalise``.

`[To do: space-time normalisation]`

In Pyroteus, space-time normalisation is achieved
through the function ``space_time_normalise``.

In many cases, it is convenient to be able to combine
different metrics. For example, if we seek to adapt
the mesh such that the value of two different error
estimators are reduced. The simplest metric combination
method from an algebraic perspective is the metric
average:

.. math::
    \tfrac12(\mathcal M_1 + \mathcal M_2),

for two metrics :math:`\mathcal M_1` and
:math:`\mathcal M_2`. Whilst mathematically simple,
the geometric interpretation of taking the metric
average is not immediately obvious. Metric intersection,
on the other hand, is geometrically straight-forward,
but non-trivial to write mathematically. The elliptic
interpretation of two metrics is the largest ellipse
which fits within the ellipses associtated with the
two input metrics. As such, metric intersection yields
a new metric whose complexity is greater than (or equal
to) its parents'. This is not true for the metric
average in general.

`[To do: citation for formula]`

Metric combination may be achieved in Pyroteus using the
functions `metric_average`, `metric_intersection`,
`metric_relaxation` (generalised average) and simply
`combine_metrics`.


Metric drivers
--------------

Pyroteus provides a number of different driver functions
for metric generation. These include ``isotropic_metric``,
which takes an error indicator field and constructs a
metric which specifies how a mesh should be adapted purely
in terms of element size (not orientation or shape).

`[To do: isotropic metric, Hessian metric]`
