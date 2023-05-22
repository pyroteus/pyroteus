.. title:: Pyroteus Goal-Oriented Mesh Adaptation Toolkit

.. only:: html

Pyroteus Goal-Oriented Mesh Adaptation Toolkit
==============================================

Pyroteus provides metric-based goal-oriented mesh adaptation
functionality to the Python-based finite element library
`Firedrake <http://www.firedrakeproject.org/>`__. The 'y' is
silent, so its pronunciation is identical to 'Proteus' - the
ancient Greek god of the constantly changing surface of the
sea.


.. rubric:: Mathematical background

Goal-oriented mesh adaptation presents one of the clearest
examples of the intersection between adjoint methods and mesh
adaptation. It is an advanced topic, so it is highly
recommended that users are familiar with adjoint methods,
mesh adaptation and the goal-oriented framework before
starting with Pyroteus.

We refer to the `Firedrake documentation
<https://firedrakeproject.org/documentation.html>`__
for an introduction to the finite element method - the
discretisation approach assumed throughout. The
`dolfin-adjoint` package (which Pyroteus uses to solve adjoint
problems) contains some `excellent documentation
<http://www.dolfin-adjoint.org/en/latest/documentation/maths/index.html>`__
on the mathematical background of adjoint problems. The
goal-oriented error estimation and metric-based mesh adaptation
functionalities provided by Pyroteus are described in the manual.

.. toctree::
    :maxdepth: 2

    Pyroteus manual <maths/index>


.. rubric:: API documentation

The classes and functions which comprise Pyroteus may be found
in the API documentation.

.. toctree::
    :maxdepth: 1

    Pyroteus API documentation <pyroteus>

They are also listed alphabetically on the :ref:`index <genindex>`
page. The index may be searched using the inbuilt
:ref:`search engine <search>`. Pyroteus source code is hosted on
`GitHub <https://github.com/pyroteus/pyroteus/>`__.


.. rubric:: Demos

Pyroteus contains a number of demos to illustrate the correct
usage of its different functionalities. It is highly recommended
that these are read in order, as many of the demos build upon
earlier ones.

*Time partitions and mesh sequences*

.. toctree::
    :maxdepth: 1

    Partitioning a time interval <demos/time_partition.py>
    Creating a mesh sequence <demos/mesh_seq.py>
    Burgers equation on a sequence of meshes <demos/burgers.py>
    Adjoint Burgers equation <demos/burgers1.py>
    Adjoint Burgers equation on two meshes <demos/burgers2.py>
    Adjoint Burgers equation with a time-integrated QoI <demos/burgers_time_integrated.py>
    Adjoint Burgers equation (object-oriented approach) <demos/burgers_oo.py>
    Solid body rotation <demos/solid_body_rotation.py>
    Solid body rotation with multiple prognostic variables <demos/solid_body_rotation_split.py>
    Advection-diffusion-reaction <demos/gray_scott.py>
    Advection-diffusion-reaction with multiple prognostic variables <demos/gray_scott_split.py>

*Error estimation*

.. toctree::
    :maxdepth: 1

    Error estimation for Burgers equation <demos/burgers_ee.py>
    Point discharge with diffusion <demos/point_discharge.py>
