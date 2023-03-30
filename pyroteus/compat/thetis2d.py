import thetis
from firedrake import *
from pyroteus.mesh_seq import MeshSeq


class FlowSolver2d(thetis.solver2d.FlowSolver2d):
    """
    Augmented version of
    :class:`thetis.solver2d.FlowSolver2d`
    class which accounts for restarting the
    simulation on a new mesh and modifying
    options prefixes.
    """

    def update_tags(self):
        """
        Modify the options prefixes which Thetis
        gives to the solvers associated with its
        timesteppers so that they match the field
        names used by Pyroteus.
        """
        self.options.swe_timestepper_options.ad_block_tag = "swe2d"
        self.options.tracer_timestepper_options.ad_block_tag = "tracer_2d"
        self.options.sediment_model_options.sediment_timestepper_options.ad_block_tag = (
            "sediment_2d"
        )
        self.options.sediment_model_options.sediment_timestepper_options.ad_block_tag = (
            "bathymetry_2d"
        )
        self.create_timestepper()
        if hasattr(self.timestepper, "timesteppers"):
            for field, ts in self.timestepper.timesteppers.items():
                self.timestepper.timesteppers[field].name = field
                self.timestepper.timesteppers[field].update_solver()
                if (
                    self.options.swe_timestepper_type != "SteadyState"
                    and self.options.tracer_timestepper_type != "SteadyState"
                ):
                    self.timestepper.timesteppers[field].solution_old.rename(
                        field + "_old"
                    )
        else:
            self.timestepper.name = "swe2d"
            self.timestepper.update_solver()
            if self.options.swe_timestepper_type != "SteadyState":
                self.timestepper.solution_old.rename("swe2d_old")

    def iterate(self, **kwargs):
        """
        Overload Thetis' 2D solve call so that it
        additionally updates the options prefixes
        associated with its solvers.
        """
        self.update_tags()
        super().iterate(**kwargs)

    def correct_counters(self, ts_data):
        """
        Adjust Thetis' 2D solver internal counters
        so that they agree with the :class:`~.MeshSeq`.
        """
        i_export = int(
            ts_data.start_time / ts_data.timestep / ts_data.timesteps_per_export
        )
        self.simulation_time = ts_data.start_time
        self.i_export = i_export
        self.next_export_t = ts_data.start_time
        self.iteration = int(ts_data.start_time / ts_data.timestep)
        self.export_initial_state = isclose(ts_data.start_time, 0.0)
        if not self.options.no_exports:
            if len(self.options.fields_to_export) > 0:
                for e in self.exporters["vtk"].exporters:
                    self.exporters["vtk"].exporters[e].set_next_export_ix(i_export)
            if len(self.options.fields_to_export_hdf5) > 0:
                for e in self.exporters["hdf5"].exporters:
                    self.exporters["hdf5"].exporters[e].set_next_export_ix(i_export)


class ThetisMeshSeq(MeshSeq):
    # TODO: doc

    def __init__(
        self,
        time_partition,
        initial_meshes,
        options,
        get_bathymetry,
        get_initial_condition,
        **kwargs,
    ):
        invalid = ("get_function_spaces", "get_form", "get_solver", "get_bcs")
        for method in invalid:
            if method in kwargs:
                raise ValueError(
                    f"'{method}' is not a valid keyword argument for {type(self)}."
                )
        super().__init__(
            time_partition,
            initial_meshes,
            get_initial_condition=get_initial_condition,
            **kwargs,
        )

        # TODO: Create solver objects
        self._solver_objs = [
            FlowSolver2d(mesh, get_bathymetry(mesh), options=options)
            for mesh in self.meshes
        ]

    def get_function_spaces(self, mesh):
        raise NotImplementedError  # TODO

    def get_form(self):
        raise NotImplementedError  # TODO

    def get_solver(self):
        raise NotImplementedError  # TODO

    def get_bcs(self):
        raise NotImplementedError(
            "'get_bcs' is not implemented because Thetis enforces boundary conditions"
            " weakly."
        )


if __name__ == "__main__":
    from thetis.options import ModelOptions2d
    from pyroteus.time_partition import TimeInstant

    mesh = UnitSquareMesh(1, 1)
    options = ModelOptions2d()

    def get_initial_condition(mesh_seq):
        fs = mesh_seq.function_spaces[0]
        return Function(fs)

    def get_bathymetry(mesh):
        fs = FunctionSpace(mesh, "CG", 1)
        return Function(fs).assign(1.0)

    mesh_seq = ThetisMeshSeq(
        TimeInstant(1), [mesh], options, get_bathymetry, get_initial_condition
    )
