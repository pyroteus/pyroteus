"""
Sequences of meshes corresponding to a :class:`~.TimePartition`.
"""
import firedrake
from firedrake.petsc import PETSc
from .interpolation import project
from .log import pyrint, debug, warning, logger, DEBUG
from .quality import get_aspect_ratios2d, get_aspect_ratios3d
from .utility import AttrDict, Mesh
from collections import OrderedDict
from collections.abc import Iterable
from functools import wraps
import numpy as np


__all__ = ["AdaptParameters", "MeshSeq"]


class AdaptParameters(AttrDict):
    """
    A class for holding parameters associated with
    adaptive mesh fixed point iteration loops.
    """

    def __init__(self, parameters={}):
        """
        :arg parameters: dictionary of parameters to set
        """
        self["miniter"] = 3  # Minimum iteration count
        self["maxiter"] = 35  # Maximum iteration count
        self["element_rtol"] = 0.001  # Relative tolerance for element count

        for key, value in parameters.items():
            if key not in self:
                raise AttributeError(f"{self} does not have {key} attribute")
            self[key] = value


class MeshSeq:
    """
    A sequence of meshes for solving a PDE associated
    with a particular :class:`~.TimePartition` of the
    temporal domain.
    """

    @PETSc.Log.EventDecorator("pyroteus.MeshSeq.__init__")
    def __init__(self, time_partition, initial_meshes, **kwargs):
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
        :kwarg warnings: print warnings?
        """
        self.time_partition = time_partition
        self.fields = time_partition.fields
        self.subintervals = time_partition.subintervals
        self.num_subintervals = time_partition.num_subintervals
        self.meshes = initial_meshes
        if not isinstance(self.meshes, Iterable):
            self.meshes = [Mesh(initial_meshes) for subinterval in self.subintervals]
        self.element_counts = [self.count_elements()]
        dim = np.array([mesh.topological_dimension() for mesh in self.meshes])
        if dim.min() != dim.max():
            raise ValueError("Meshes must all have the same topological dimension")
        self.dim = dim.min()
        if logger.level == DEBUG:
            for i, mesh in enumerate(self.meshes):
                nc = mesh.num_cells()
                nv = mesh.num_vertices()
                if self.dim == 2:
                    ar = get_aspect_ratios2d(mesh)
                else:
                    ar = get_aspect_ratios3d(mesh)
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
        self.params = kwargs.get("parameters", AdaptParameters())
        self.warn = kwargs.get("warnings", True)
        self._lagged_dep_idx = {}
        self.sections = [{} for mesh in self]
        if not hasattr(self, "steady"):
            self.steady = False

    def debug(self, msg):
        debug(f"MeshSeq: {msg}")

    def warning(self, msg):
        warning(f"MeshSeq: {msg}")

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, i):
        return self.meshes[i]

    def __setitem__(self, i, mesh):
        self.meshes[i] = mesh

    def count_elements(self):
        return [mesh.num_cells() for mesh in self]  # TODO: make parallel safe

    def plot(self, fig=None, axes=None, **kwargs):
        """
        Plot the meshes comprising a 2D :class:`~.MeshSeq`.

        :kwarg fig: matplotlib figure
        :kwarg axes: matplotlib axes
        :kwargs: parameters to pass to :func:`firedrake.plot.triplot`
            function
        :return: matplotlib figure and axes for the plots
        """
        if self.dim != 2:
            raise ValueError("MeshSeq plotting only supported in 2D")
        kwargs.setdefault("interior_kw", {"edgecolor": "k"})
        kwargs.setdefault("boundary_kw", {"edgecolor": "k"})
        if fig is None and axes is None:
            from matplotlib.pyplot import subplots

            n = len(self)
            size = (5 * n, 5)
            fig, axes = subplots(ncols=n, nrows=1, figsize=size)
        i = 0
        for axis in axes:
            if not isinstance(axis, Iterable):
                axis = [axis]
            for ax in axis:
                ax.set_title(f"MeshSeq[{i}]")
                firedrake.triplot(self.meshes[i], axes=ax, **kwargs)
                ax.axis(False)
                i += 1
        return fig, axes

    def get_function_spaces(self, mesh):
        if self._get_function_spaces is None:
            raise NotImplementedError("get_function_spaces needs implementing")
        return self._get_function_spaces(mesh)

    def get_initial_condition(self):
        if self._get_initial_condition is not None:
            return self._get_initial_condition(self)
        return {
            field: firedrake.Function(fs[0])
            for field, fs in self.function_spaces.items()
        }

    def get_form(self):
        if self._get_form is None:
            raise NotImplementedError("get_form needs implementing")
        return self._get_form(self)

    def get_solver(self):
        if self._get_solver is None:
            raise NotImplementedError("get_solver needs implementing")
        return self._get_solver(self)

    def get_bcs(self):
        if self._get_bcs is not None:
            return self._get_bcs(self)

    @property
    def _function_spaces_consistent(self):
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
    def function_spaces(self):
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
    def initial_condition(self):
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
    def form(self):
        return self.get_form()

    @property
    def solver(self):
        return self.get_solver()

    @property
    def bcs(self):
        return self.get_bcs()

    @PETSc.Log.EventDecorator("pyroteus.MeshSeq.get_checkpoints")
    def get_checkpoints(self, solver_kwargs={}):
        """
        Solve forward on the sequence of meshes,
        extracting checkpoints corresponding to
        the starting fields on each subinterval.

        :kwarg solver_kwargs: additional keyword
            arguments which will be passed to
            the solver
        """
        solver = self.solver
        checkpoints = [self.initial_condition]
        for i in range(len(self)):
            sols = solver(
                self, checkpoints[i], *self.time_partition[i], **solver_kwargs
            )
            assert issubclass(sols.__class__, dict), "solver should return a dict"
            assert set(self.fields).issubclass(
                set(sols.keys())
            ), "missing fields from solver"
            assert set(sols.keys()).issubclass(
                set(self.fields)
            ), "more solver outputs than fields"
            if i < len(self) - 1:
                checkpoints.append(
                    {
                        field: project(sols[field], fs[i + 1])
                        for field, fs in self._fs.items()
                    }
                )
        return checkpoints

    def get_solve_blocks(self, field, subinterval=0, has_adj_sol=True):
        """
        Get all blocks of the tape corresponding to
        solve steps for prognostic solution ``field``
        on a given ``subinterval``.
        """
        from firedrake.adjoint.solving import get_solve_blocks
        from pyadjoint import get_working_tape

        # Get all blocks
        blocks = get_working_tape().get_blocks()
        if len(blocks) == 0:
            self.warning("Tape has no blocks!")
            return blocks

        # Restrict to solve blocks
        solve_blocks = get_solve_blocks()
        if len(solve_blocks) == 0:
            self.warning("Tape has no solve blocks!")
            return solve_blocks

        # Slice solve blocks by field
        solve_blocks = [
            block
            for block in solve_blocks
            if block.tag is not None and field in block.tag
        ]
        if len(solve_blocks) == 0:
            self.warning(
                f"No solve blocks associated with field '{field}'.\n"
                "Has ad_block_tag been used correctly?"
            )
            return solve_blocks
        self.debug(
            f"Field '{field}' on subinterval {subinterval} has {len(solve_blocks)} solve blocks"
        )

        # Default adjoint solution to zero, rather than None
        if has_adj_sol:
            if all(block.adj_sol is None for block in solve_blocks):
                self.warning(
                    "No block has an adjoint solution. Has the adjoint equation been solved?"
                )
            for block in solve_blocks:
                if block.adj_sol is None:
                    block.adj_sol = firedrake.Function(
                        self.function_spaces[field][subinterval], name=field
                    )

        # Check FunctionSpaces are consistent across solve blocks
        element = solve_blocks[0].function_space.ufl_element()
        for block in solve_blocks:
            if element != block.function_space.ufl_element():
                raise ValueError(
                    f"Solve block list for field {field} contains mismatching elements"
                    f" ({element} vs. {block.function_space.ufl_element()})"
                )

        # Check the number of timesteps divides the number of solve blocks
        num_timesteps = self.time_partition[subinterval].num_timesteps
        ratio = len(solve_blocks) / num_timesteps
        if not np.isclose(np.round(ratio), ratio):
            raise ValueError(
                f"Number of timesteps for field '{field}' does not divide number of solve"
                f" blocks ({num_timesteps} vs. {len(solve_blocks)}). If you are trying to"
                " use a multi-stage Runge-Kutta method, then this is not supported."
            )
        return solve_blocks

    def get_lagged_dependency_index(self, field, subinterval, solve_blocks):
        """
        Get the dependency index corresponding
        to the lagged forward solution for a
        solve block.

        :arg field: field of interest
        :arg subinterval: subinterval index
        :arg solve_blocks: list of taped
            :class:`firedrake.adjoint.blocks.GenericSolveBlocks`
        """
        if field in self._lagged_dep_idx:
            return self._lagged_dep_idx[field]
        fs = self.function_spaces[field][subinterval]
        fwd_old_idx = [
            dep_index
            for dep_index, dep in enumerate(solve_blocks[0]._dependencies)
            if hasattr(dep.output, "function_space")
            and dep.output.function_space() == solve_blocks[0].function_space == fs
            and hasattr(dep.output, "name")
            and dep.output.name() == field + "_old"
        ]
        if len(fwd_old_idx) == 0:
            if not self.steady:
                self.warning(
                    f"Solve block for field '{field}' on"
                    f" subinterval {subinterval} has no dependencies"
                )
            fwd_old_idx = None
        else:
            if len(fwd_old_idx) > 1:
                self.warning(
                    f"Solve block for field '{field}' on subinterval"
                    f" {subinterval} has dependencies in the prognostic"
                    " space other than the PDE solution at the previous"
                    f" timestep (dep. indices {fwd_old_idx}). Naively"
                    " assuming the first to be the right one."
                )
            fwd_old_idx = fwd_old_idx[0]
        self._lagged_dep_idx[field] = fwd_old_idx
        return fwd_old_idx

    @PETSc.Log.EventDecorator("pyroteus.MeshSeq.solve_forward")
    def solve_forward(self, solver_kwargs={}, clear_tape=True):
        """
        Solve a forward problem on a sequence of subintervals.

        A dictionary of solution fields is computed, the contents
        of which give values at all exported timesteps, indexed
        first by the field label and then by type. The contents
        of these nested dictionaries are lists which are indexed
        first by subinterval and then by export. For a given
        exported timestep, the solution types are:

        * ``'forward'``: the forward solution after taking the
            timestep;
        * ``'forward_old'``: the forward solution before taking
            the timestep.

        :kwarg solver_kwargs: a dictionary providing parameters
            to the solver. Any keyword arguments for the QoI
            should be included as a subdict with label 'qoi_kwargs'
        :kwarg clear_tape: should the tape be cleared at the end
            of an iteration?

        :return solution: an :class:`~.AttrDict` containing
            solution fields and their lagged versions.
        """
        from firedrake_adjoint import pyadjoint

        num_subintervals = len(self)
        function_spaces = self.function_spaces
        P = self.time_partition

        # Create arrays to hold exported forward solutions
        solutions = AttrDict(
            {
                field: AttrDict(
                    {
                        label: [
                            [
                                firedrake.Function(fs, name=f"{field}_{label}")
                                for j in range(P.exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(function_spaces[field])
                        ]
                        for label in ("forward", "forward_old")
                    }
                )
                for field in self.fields
            }
        )

        # Wrap solver to extract controls
        solver = self.solver

        @PETSc.Log.EventDecorator("pyroteus.MeshSeq.solve_adjoint.evaluate_fwd")
        @wraps(solver)
        def wrapped_solver(i, ic, **kwargs):
            init = AttrDict(
                {field: ic[field].copy(deepcopy=True) for field in self.fields}
            )
            self.controls = [pyadjoint.Control(init[field]) for field in self.fields]
            out = solver(i, init, **kwargs)
            if i < len(self) - 1:
                for field, fs in function_spaces.items():
                    out[field] = project(out[field], fs[i + 1])
            return out

        # Clear tape
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()

        checkpoint = self.initial_condition
        for i in range(num_subintervals):
            stride = P.timesteps_per_export[i]
            num_exports = P.exports_per_subinterval[i]

            # Annotate tape on current subinterval
            checkpoint = wrapped_solver(i, checkpoint, **solver_kwargs)

            # Loop over prognostic variables
            for field, fs in function_spaces.items():

                # Get solve blocks
                solve_blocks = self.get_solve_blocks(
                    field, subinterval=i, has_adj_sol=False
                )
                num_solve_blocks = len(solve_blocks)
                assert num_solve_blocks > 0, (
                    "Looks like no solves were written to tape!"
                    " Does the solution depend on the initial condition?"
                )
                if "forward_old" in solutions[field]:
                    fwd_old_idx = self.get_lagged_dependency_index(
                        field, i, solve_blocks
                    )
                else:
                    fwd_old_idx = None
                if fwd_old_idx is None and "forward_old" in solutions[field]:
                    solutions[field].pop("forward_old")

                # Extract solution data
                sols = solutions[field]
                if len(solve_blocks[::stride]) >= num_exports:
                    raise ValueError(
                        f"More solve blocks than expected"
                        f" ({len(solve_blocks[::stride])} > {num_exports-1})"
                    )
                for j, block in zip(range(num_exports - 1), solve_blocks[::stride]):
                    if fwd_old_idx is not None:
                        dep = block._dependencies[fwd_old_idx]
                        sols.forward_old[i][j].assign(dep.saved_output)
                    sols.forward[i][j].assign(block._outputs[0].saved_output)

            # Clear tape
            if clear_tape:
                tape.clear_tape()

        return solutions

    def check_element_count_convergence(self):
        """
        Check for convergence of the fixed point iteration
        due to the relative difference in element count being
        smaller than the specified tolerance.

        :return: ``True`` if converged, else ``False``
        """
        P = self.params
        self.converged = False
        if len(self.element_counts) < max(2, P.miniter + 1):
            return
        self.converged = True
        elems_ = self.element_counts[-2]
        elems = self.element_counts[-1]
        for ne_, ne in zip(elems_, elems):
            if abs(ne - ne_) > P.element_rtol * ne_:
                self.converged = False

    @PETSc.Log.EventDecorator("pyroteus.MeshSeq.fixed_point_iteration")
    def fixed_point_iteration(self, adaptor, update_params=None, solver_kwargs={}):
        r"""
        Apply goal-oriented mesh adaptation using
        a fixed point iteration loop.

        :arg adaptor: function for adapting the mesh sequence.
            Its arguments are the :class:`~.MeshSeq` instance and
            the dictionary of solution
            :class:`firedrake.function.Function`\s
        :kwarg update_params: function for updating
            :attr:`~.MeshSeq.params` at each iteration. Its
            arguments are the parameter class and the fixed point
            iteration
        :kwarg solver_kwargs: a dictionary providing parameters
            to the solver
        """
        P = self.params
        self.element_counts = [self.count_elements()]
        self.converged = False
        for fp_iteration in range(P.maxiter):
            if update_params is not None:
                update_params(P, fp_iteration)

            # Solve the forward problem over all meshes
            sols = self.solve_forward(solver_kwargs=solver_kwargs)

            # Adapt meshes and log element counts
            adaptor(self, sols)
            self.element_counts.append(self.count_elements())

            # Check for element count convergence
            self.check_element_count_convergence()
            if self.converged:
                pyrint(
                    "Terminated due to element count convergence"
                    f" after {fp_iteration+1} iterations"
                )
                break
        if not self.converged:
            pyrint(f"Failed to converge in {P.maxiter} iterations")
