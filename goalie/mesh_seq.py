"""
Sequences of meshes corresponding to a :class:`~.TimePartition`.
"""
import firedrake
from firedrake.petsc import PETSc
from firedrake.adjoint_utils.solving import get_solve_blocks
from firedrake.adjoint import pyadjoint
from .interpolation import project
from .log import pyrint, debug, warning, info, logger, DEBUG
from .options import AdaptParameters
from animate.quality import QualityMeasure
from .time_partition import TimePartition
from .utility import AttrDict, Mesh
from collections import OrderedDict
from collections.abc import Callable, Iterable
import matplotlib
import numpy as np
from typing import Tuple


__all__ = ["MeshSeq"]


class MeshSeq:
    """
    A sequence of meshes for solving a PDE associated
    with a particular :class:`~.TimePartition` of the
    temporal domain.
    """

    @PETSc.Log.EventDecorator("goalie.MeshSeq.__init__")
    def __init__(self, time_partition: TimePartition, initial_meshes: list, **kwargs):
        r"""
        :arg time_partition: the :class:`~.TimePartition` which
            partitions the temporal domain
        :arg initial_meshes: list of meshes corresponding to
            the subintervals of the :class:`~.TimePartition`,
            or a single mesh to use for all subintervals
        :kwarg get_function_spaces: a function, whose only
            argument is a :class:`~.MeshSeq`, which constructs
            prognostic
            :class:`firedrake.functionspaceimpl.FunctionSpace`\s
            for each subinterval
        :kwarg get_initial_condition: a function, whose only
            argument is a :class:`~.MeshSeq`, which specifies
            initial conditions on the first mesh
        :kwarg get_form: a function, whose only argument is a
            :class:`~.MeshSeq`, which returns a function that
            generates the PDE weak form
        :kwarg get_solver: a function, whose only argument is
            a :class:`~.MeshSeq`, which returns a function
            that integrates initial data over a subinterval
        :kwarg get_bcs: a function, whose only argument is a
            :class:`~.MeshSeq`, which returns a function that
            determines any Dirichlet boundary conditions
        :kwarg parameters: :class:`~.AdaptParameters` instance
        """
        self.time_partition = time_partition
        self.fields = time_partition.fields
        self.field_types = {
            field: field_type
            for field, field_type in zip(self.fields, time_partition.field_types)
        }
        self.subintervals = time_partition.subintervals
        self.num_subintervals = time_partition.num_subintervals
        self.meshes = initial_meshes
        if not isinstance(self.meshes, Iterable):
            self.meshes = [Mesh(initial_meshes) for subinterval in self.subintervals]
        self.element_counts = [self.count_elements()]
        self.vertex_counts = [self.count_vertices()]
        dim = np.array([mesh.topological_dimension() for mesh in self.meshes])
        if dim.min() != dim.max():
            raise ValueError("Meshes must all have the same topological dimension.")
        self.dim = dim.min()
        if logger.level == DEBUG:
            for i, mesh in enumerate(self.meshes):
                nc = mesh.num_cells()
                nv = mesh.num_vertices()
                qm = QualityMeasure(mesh)
                ar = qm("aspect_ratio")
                mar = ar.vector().gather().max()
                self.debug(
                    f"{i}: {nc:7d} cells, {nv:7d} vertices,   max aspect ratio {mar:.2f}"
                )
            debug(100 * "-")
        self._fs = None
        self._get_function_spaces = kwargs.get("get_function_spaces")
        self._get_initial_condition = kwargs.get("get_initial_condition")
        self._get_form = kwargs.get("get_form")
        self._get_solver = kwargs.get("get_solver")
        self._get_bcs = kwargs.get("get_bcs")
        self.params = kwargs.get("parameters")
        self.steady = time_partition.steady
        self.check_convergence = np.array([True] * len(self), dtype=bool)
        self.converged = np.array([False] * len(self), dtype=bool)
        self.fp_iteration = 0
        if self.params is None:
            self.params = AdaptParameters()
        self.sections = [{} for mesh in self]

    def __str__(self) -> str:
        return f"{[str(mesh) for mesh in self.meshes]}"

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if len(self) == 1:
            return f"{name}([{repr(self.meshes[0])}])"
        elif len(self) == 2:
            return f"{name}([{repr(self.meshes[0])}, {repr(self.meshes[1])}])"
        else:
            return f"{name}([{repr(self.meshes[0])}, ..., {repr(self.meshes[-1])}])"

    def debug(self, msg: str):
        debug(f"{self.__class__.__name__}: {msg}")

    def warning(self, msg: str):
        warning(f"{self.__class__.__name__}: {msg}")

    def info(self, msg: str):
        info(f"{self.__class__.__name__}: {msg}")

    def __len__(self) -> int:
        return len(self.meshes)

    def __getitem__(self, i: int) -> firedrake.MeshGeometry:
        return self.meshes[i]

    def __setitem__(
        self, i: int, mesh: firedrake.MeshGeometry
    ) -> firedrake.MeshGeometry:
        self.meshes[i] = mesh

    def count_elements(self) -> list:
        return [mesh.num_cells() for mesh in self]  # TODO: make parallel safe

    def count_vertices(self) -> list:
        return [mesh.num_vertices() for mesh in self]  # TODO: make parallel safe

    def plot(
        self, **kwargs
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        """
        Plot the meshes comprising a 2D :class:`~.MeshSeq`.

        :kwarg fig: matplotlib figure
        :kwarg axes: matplotlib axes
        :kwargs: parameters to pass to :func:`firedrake.plot.triplot`
            function
        :return: matplotlib figure and axes for the plots
        """
        from matplotlib.pyplot import subplots

        if self.dim != 2:
            raise ValueError("MeshSeq plotting only supported in 2D")

        # Process kwargs
        fig = kwargs.pop("fig", None)
        axes = kwargs.pop("axes", None)
        interior_kw = {"edgecolor": "k"}
        interior_kw.update(kwargs.pop("interior_kw", {}))
        boundary_kw = {"edgecolor": "k"}
        boundary_kw.update(kwargs.pop("boundary_kw", {}))
        kwargs["interior_kw"] = interior_kw
        kwargs["boundary_kw"] = boundary_kw
        if fig is None or axes is None:
            n = len(self)
            fig, axes = subplots(ncols=n, nrows=1, figsize=(5 * n, 5))

        # Loop over all axes and plot the meshes
        k = 0
        if not isinstance(axes, Iterable):
            axes = [axes]
        for i, axis in enumerate(axes):
            if not isinstance(axis, Iterable):
                axis = [axis]
            for ax in axis:
                ax.set_title(f"MeshSeq[{k}]")
                firedrake.triplot(self.meshes[k], axes=ax, **kwargs)
                ax.axis(False)
                k += 1
            if len(axis) == 1:
                axes[i] = axis[0]
        if len(axes) == 1:
            axes = axes[0]
        return fig, axes

    def get_function_spaces(self, mesh: firedrake.MeshGeometry) -> Callable:
        if self._get_function_spaces is None:
            raise NotImplementedError("'get_function_spaces' needs implementing.")
        return self._get_function_spaces(mesh)

    def get_initial_condition(self) -> dict:
        if self._get_initial_condition is not None:
            return self._get_initial_condition(self)
        return {
            field: firedrake.Function(fs[0])
            for field, fs in self.function_spaces.items()
        }

    def get_form(self) -> Callable:
        if self._get_form is None:
            raise NotImplementedError("'get_form' needs implementing.")
        return self._get_form(self)

    def get_solver(self) -> Callable:
        if self._get_solver is None:
            raise NotImplementedError("'get_solver' needs implementing.")
        return self._get_solver(self)

    def get_bcs(self) -> Callable:
        if self._get_bcs is not None:
            return self._get_bcs(self)

    @property
    def _function_spaces_consistent(self) -> bool:
        consistent = len(self.time_partition) == len(self)
        consistent &= all(len(self) == len(self._fs[field]) for field in self.fields)
        for field in self.fields:
            consistent &= all(
                mesh == fs.mesh() for mesh, fs in zip(self.meshes, self._fs[field])
            )
            consistent &= all(
                self._fs[field][0].ufl_element() == fs.ufl_element()
                for fs in self._fs[field]
            )
        return consistent

    @property
    def function_spaces(self) -> list:
        if self._fs is None or not self._function_spaces_consistent:
            self._fs = [self.get_function_spaces(mesh) for mesh in self.meshes]
            self._fs = AttrDict(
                {
                    field: [self._fs[i][field] for i in range(len(self))]
                    for field in self.fields
                }
            )
        assert (
            self._function_spaces_consistent
        ), "Meshes and function spaces are inconsistent"
        return self._fs

    @property
    def initial_condition(self) -> AttrDict:
        ic = OrderedDict(self.get_initial_condition())
        assert issubclass(
            ic.__class__, dict
        ), "`get_initial_condition` should return a dict"
        assert set(self.fields).issubset(
            set(ic.keys())
        ), "missing fields in initial condition"
        assert set(ic.keys()).issubset(
            set(self.fields)
        ), "more initial conditions than fields"
        return AttrDict(ic)

    @property
    def form(self) -> Callable:
        return self.get_form()

    @property
    def solver(self) -> Callable:
        return self.get_solver()

    @property
    def bcs(self) -> Callable:
        return self.get_bcs()

    @PETSc.Log.EventDecorator()
    def get_checkpoints(
        self, solver_kwargs: dict = {}, run_final_subinterval: bool = False
    ) -> list:
        """
        Solve forward on the sequence of meshes, extracting checkpoints corresponding
        to the starting fields on each subinterval.

        :kwarg solver_kwargs: additional keyword arguments which will be passed to the
            solver
        :kwarg run_final_subinterval: toggle whether to solve the PDE on the final
            subinterval
        """
        N = len(self)

        # The first checkpoint is the initial condition
        checkpoints = [self.initial_condition]

        # If there is only one subinterval then we are done
        if N == 1 and not run_final_subinterval:
            return checkpoints

        # Otherwise, solve each subsequent subinterval, in each case making use of the
        # previous checkpoint
        for i in range(N if run_final_subinterval else N - 1):
            sols = self.solver(i, checkpoints[i], **solver_kwargs)
            if not isinstance(sols, dict):
                raise TypeError(
                    f"Solver should return a dictionary, not '{type(sols)}'."
                )

            # Check that the output of the solver is as expected
            fields = set(sols.keys())
            if not set(self.fields).issubset(fields):
                diff = set(self.fields).difference(fields)
                raise ValueError(f"Fields are missing from the solver: {diff}.")
            if not fields.issubset(set(self.fields)):
                diff = fields.difference(set(self.fields))
                raise ValueError(f"Unexpected solver outputs: {diff}.")

            # Transfer between meshes using conservative projection
            if i < N - 1:
                checkpoints.append(
                    AttrDict(
                        {
                            field: project(sols[field], fs[i + 1])
                            for field, fs in self._fs.items()
                        }
                    )
                )

        return checkpoints

    @PETSc.Log.EventDecorator()
    def get_solve_blocks(self, field: str, subinterval: int) -> list:
        """
        Get all blocks of the tape corresponding to solve steps for prognostic solution
        field on a given subinterval.

        :arg field: name of the prognostic solution field
        :arg subinterval: subinterval index
        """
        tape = pyadjoint.get_working_tape()
        if tape is None:
            self.warning("Tape does not exist!")
            return []

        blocks = tape.get_blocks()
        if len(blocks) == 0:
            self.warning("Tape has no blocks!")
            return blocks

        # Restrict to solve blocks
        solve_blocks = get_solve_blocks()
        if len(solve_blocks) == 0:
            self.warning("Tape has no solve blocks!")
            return solve_blocks

        # Select solve blocks whose tags correspond to the field name
        solve_blocks = [
            block
            for block in solve_blocks
            if isinstance(block.tag, str) and block.tag.startswith(field)
        ]
        N = len(solve_blocks)
        if N == 0:
            self.warning(
                f"No solve blocks associated with field '{field}'."
                " Has ad_block_tag been used correctly?"
            )
            return solve_blocks
        self.debug(
            f"Field '{field}' on subinterval {subinterval} has {N} solve blocks."
        )

        # Check FunctionSpaces are consistent across solve blocks
        element = self.function_spaces[field][subinterval].ufl_element()
        for block in solve_blocks:
            if element != block.function_space.ufl_element():
                raise ValueError(
                    f"Solve block list for field '{field}' contains mismatching elements:"
                    f" {element} vs. {block.function_space.ufl_element()}."
                )

        # Check that the number of timesteps does not exceed the number of solve blocks
        num_timesteps = self.time_partition[subinterval].num_timesteps
        if num_timesteps > N:
            raise ValueError(
                f"Number of timesteps exceeds number of solve blocks for field '{field}'"
                f" on subinterval {subinterval}: {num_timesteps} > {N}."
            )

        # Check the number of timesteps is divisible by the number of solve blocks
        ratio = num_timesteps / N
        if not np.isclose(np.round(ratio), ratio):
            raise ValueError(
                "Number of timesteps is not divisible by number of solve blocks for"
                f" field '{field}' on subinterval {subinterval}: {num_timesteps} vs."
                f" {N}."
            )
        return solve_blocks

    def _output(self, field, subinterval, solve_block):
        """
        For a given solve block and solution field, get the block's outputs which
        corresponds to the solution from the current timestep.

        :arg field: field of interest
        :arg subinterval: subinterval index
        :arg solve_block: taped :class:`firedrake.adjoint.blocks.GenericSolveBlock`
        """
        fs = self.function_spaces[field][subinterval]

        # Loop through the solve block's outputs
        candidates = []
        for out in solve_block._outputs:
            # Look for Functions with matching function spaces
            if not isinstance(out.output, firedrake.Function):
                continue
            if out.output.function_space() != fs:
                continue

            # Look for Functions whose name matches that of the field
            # NOTE: Here we assume that the user has set this correctly in their
            #       get_solver method
            if not out.output.name() == field:
                continue

            # Add to the list of candidates
            candidates.append(out)

        # Check for existence and uniqueness
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise AttributeError(
                "Cannot determine a unique output index for the solution associated"
                f" with field '{field}' out of {len(candidates)} candidates."
            )
        elif not self.steady:
            raise AttributeError(
                f"Solve block for field '{field}' on subinterval {subinterval} has no"
                " outputs."
            )

    def _dependency(self, field, subinterval, solve_block):
        """
        For a given solve block and solution field, get the block's dependency which
        corresponds to the solution from the previous timestep.

        :arg field: field of interest
        :arg subinterval: subinterval index
        :arg solve_block: taped :class:`firedrake.adjoint.blocks.GenericSolveBlock`
        """
        if self.field_types[field] == "steady":
            return
        fs = self.function_spaces[field][subinterval]

        # Loop through the solve block's dependencies
        candidates = []
        for dep in solve_block._dependencies:
            # Look for Functions with matching function spaces
            if not isinstance(dep.output, firedrake.Function):
                continue
            if dep.output.function_space() != fs:
                continue

            # Look for Functions whose name is the lagged version of the field's
            # NOTE: Here we assume that the user has set this correctly in their
            #       get_solver method
            if not dep.output.name() == f"{field}_old":
                continue

            # Add to the list of candidates
            candidates.append(dep)

        # Check for existence and uniqueness
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise AttributeError(
                "Cannot determine a unique dependency index for the lagged solution"
                f" associated with field '{field}' out of {len(candidates)} candidates."
            )
        elif not self.steady:
            raise AttributeError(
                f"Solve block for field '{field}' on subinterval {subinterval} has no"
                " dependencies."
            )

    @PETSc.Log.EventDecorator()
    def solve_forward(self, solver_kwargs: dict = {}) -> AttrDict:
        """
        Solve a forward problem on a sequence of subintervals.

        A dictionary of solution fields is computed, the contents of which give values
        at all exported timesteps, indexed first by the field label and then by type.
        The contents of these nested dictionaries are nested lists which are indexed
        first by subinterval and then by export. For a given exported timestep, the
        solution types are:

        * ``'forward'``: the forward solution after taking the
            timestep;
        * ``'forward_old'``: the forward solution before taking
            the timestep.

        :kwarg solver_kwargs: a dictionary providing parameters to the solver. Any
            keyword arguments for the QoI should be included as a subdict with label
            'qoi_kwargs'

        :return solution: an :class:`~.AttrDict` containing solution fields and their
            lagged versions.
        """
        num_subintervals = len(self)
        function_spaces = self.function_spaces
        P = self.time_partition
        solver = self.solver

        # Create arrays to hold exported forward solutions and their lagged
        # counterparts
        labels = ("forward", "forward_old")
        solutions = AttrDict(
            {
                field: AttrDict(
                    {
                        label: [
                            [
                                firedrake.Function(fs, name=f"{field}_{label}")
                                for j in range(P.num_exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(function_spaces[field])
                        ]
                        for label in labels
                    }
                )
                for field in self.fields
            }
        )

        # Start annotating
        if pyadjoint.annotate_tape():
            tape = pyadjoint.get_working_tape()
            if tape is not None:
                tape.clear_tape()
        else:
            pyadjoint.continue_annotation()

        # Loop over the subintervals
        checkpoint = self.initial_condition
        for i in range(num_subintervals):
            stride = P.num_timesteps_per_export[i]
            num_exports = P.num_exports_per_subinterval[i]

            # Annotate tape on current subinterval
            checkpoint = solver(i, checkpoint, **solver_kwargs)

            # Loop over prognostic variables
            for field, fs in function_spaces.items():
                # Get solve blocks
                solve_blocks = self.get_solve_blocks(field, i)
                num_solve_blocks = len(solve_blocks)
                if num_solve_blocks == 0:
                    raise ValueError(
                        "Looks like no solves were written to tape!"
                        " Does the solution depend on the initial condition?"
                    )
                if fs[0].ufl_element() != solve_blocks[0].function_space.ufl_element():
                    raise ValueError(
                        f"Solve block list for field '{field}' contains mismatching"
                        f" finite elements: ({fs[0].ufl_element()} vs. "
                        f" {solve_blocks[0].function_space.ufl_element()})"
                    )

                # Extract solution data
                if len(solve_blocks[::stride]) >= num_exports:
                    raise ValueError(
                        f"More solve blocks than expected"
                        f" ({len(solve_blocks[::stride])} > {num_exports-1})"
                    )

                # Update solution data based on block dependencies and outputs
                sols = solutions[field]
                for j, block in zip(range(num_exports - 1), solve_blocks[::stride]):
                    # Current solution is determined from outputs
                    out = self._output(field, i, block)
                    if out is not None:
                        sols.forward[i][j].assign(out.saved_output)

                    # Lagged solution comes from dependencies
                    dep = self._dependency(field, i, block)
                    if dep is not None:
                        sols.forward_old[i][j].assign(dep.saved_output)

            # Transfer the checkpoint between subintervals
            if i < num_subintervals - 1:
                checkpoint = AttrDict(
                    {
                        field: project(checkpoint[field], fs[i + 1])
                        for field, fs in self._fs.items()
                    }
                )

            # Clear the tape to reduce the memory footprint
            pyadjoint.get_working_tape().clear_tape()

        return solutions

    def check_element_count_convergence(self):
        """
        Check for convergence of the fixed point iteration due to the relative
        difference in element count being smaller than the specified tolerance.

        :return: Boolean array with ``True`` in the appropriate entry if convergence is
            detected on a subinterval.
        """
        if self.params.drop_out_converged:
            converged = self.converged
        else:
            converged = np.array([False] * len(self), dtype=bool)
        if len(self.element_counts) >= max(2, self.params.miniter + 1):
            for i, (ne_, ne) in enumerate(zip(*self.element_counts[-2:])):
                if not self.check_convergence[i]:
                    self.info(
                        f"Skipping element count convergence check on subinterval {i})"
                        f" because check_convergence[{i}] == False."
                    )
                    continue
                if abs(ne - ne_) <= self.params.element_rtol * ne_:
                    converged[i] = True
                    if len(self) == 1:
                        pyrint(
                            f"Element count converged after {self.fp_iteration+1}"
                            " iterations under relative tolerance"
                            f" {self.params.element_rtol}."
                        )
                    else:
                        pyrint(
                            f"Element count converged on subinterval {i} after"
                            f" {self.fp_iteration+1} iterations under relative tolerance"
                            f" {self.params.element_rtol}."
                        )

        # Check only early subintervals are marked as converged
        if self.params.drop_out_converged and not converged.all():
            first_not_converged = converged.argsort()[0]
            converged[first_not_converged:] = False

        return converged

    @PETSc.Log.EventDecorator()
    def fixed_point_iteration(
        self, adaptor: Callable, solver_kwargs: dict = {}, **kwargs
    ):
        r"""
        Apply goal-oriented mesh adaptation using a fixed point iteration loop.

        :arg adaptor: function for adapting the mesh sequence. Its arguments are the
            :class:`~.MeshSeq` instance and the dictionary of solution
            :class:`firedrake.function.Function`\s. It should return ``True`` if the
            convergence criteria checks are to be skipped for this iteration. Otherwise,
            it should return ``False``.
        :kwarg update_params: function for updating :attr:`~.MeshSeq.params` at each
            iteration. Its arguments are the parameter class and the fixed point
            iteration
        :kwarg solver_kwargs: a dictionary providing parameters to the solver
        """
        update_params = kwargs.get("update_params")
        self.element_counts = [self.count_elements()]
        self.vertex_counts = [self.count_vertices()]
        self.converged[:] = False
        self.check_convergence[:] = True

        for self.fp_iteration in range(self.params.maxiter):
            if update_params is not None:
                update_params(self.params, self.fp_iteration)

            # Solve the forward problem over all meshes
            sols = self.solve_forward(solver_kwargs=solver_kwargs)

            # Adapt meshes, logging element and vertex counts
            continue_unconditionally = adaptor(self, sols)
            if self.params.drop_out_converged:
                self.check_convergence[:] = np.logical_not(
                    np.logical_or(continue_unconditionally, self.converged)
                )
            self.element_counts.append(self.count_elements())
            self.vertex_counts.append(self.count_vertices())

            # Check for element count convergence
            self.converged[:] = self.check_element_count_convergence()
            if self.converged.all():
                break
        else:
            for i, conv in enumerate(self.converged):
                if not conv:
                    pyrint(
                        f"Failed to converge on subinterval {i} in"
                        f" {self.params.maxiter} iterations."
                    )

        return sols
