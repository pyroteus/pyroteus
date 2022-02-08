try:
    from thetis import *
except ImportError:
    raise ImportError("Thetis is not installed!")
from numpy import isclose


class FlowSolver2d(thetis.solver2d.FlowSolver2d):
    """
    Augmented version of Thetis' ``FlowSolver2d``
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
        super(FlowSolver2d, self).iterate(**kwargs)

    def correct_counters(self, ts_data):
        """
        Adjust Thetis' 2D solver internal counters
        so that they agree with the :class:`MeshSeq`.
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
