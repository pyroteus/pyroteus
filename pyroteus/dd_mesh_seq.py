from .go_mesh_seq import GoalOrientedMeshSeq
from .time_partition import TimePartition
from .utility import AttrDict
from firedrake import Function, FunctionSpace
from firedrake.petsc import PETSc
import numpy as np
import torch
from typing import Tuple


class GoalOrientedDataGenerator(GoalOrientedMeshSeq):
    """
    A :class:`MeshSeq` subclass, which generates data to be used to train a
    :class:`DataDrivenMeshSeq`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if PETSc.COMM_WORLD.size > 1:
            raise NotImplementedError(
                "GoalOrientedDataGenerator is only currently implemented in serial."
            )  # TODO: parallel case

    @PETSc.Log.EventDecorator()
    def collect_features(
        self, solutions: AttrDict, field: str, subinterval: int
    ) -> np.array:
        """
        Extract feature data from forward and adjoint solution fields over each element.

        :arg solutions: :class:`AttrDict` of solution fields
        :arg field: label of the field to collect data for
        :arg subinterval: subinterval of the :class:`MeshSeq` to collect data for
        """
        # fwd_sols = solutions[field]["forward"][subinterval]
        # adj_sols = solutions[field]["adjoint"][subinterval]
        raise NotImplementedError  # TODO: Extract NumPy arrays for the whole subinterval

    def collect_targets(
        self, indicators: AttrDict, field: str, subinterval: int
    ) -> np.array:
        """
        Extract target data from goal-oriented error indicator fields over each element.

        :arg indicators: :class:`AttrDict` of indicator fields.
        :arg field: label of the field to collect data for
        :arg subinterval: subinterval of the :class:`MeshSeq` to collect data for
        """
        # indi = indicators[field][subinterval]
        raise NotImplementedError  # TODO: Extract NumPy arrays for the whole subinterval

    @staticmethod
    def concat(a, b):
        return b if a is None else np.concatenate((a, b), axis=0)

    @PETSc.Log.EventDecorator()
    def generate_data(self, **kwargs) -> Tuple(AttrDict, AttrDict, np.array, np.array):
        """
        Generate data for training by computing goal-oriented error estimates using a
        standard method.

        Keyword arguments passed to :meth:`GoalOrientedMeshSeq.indicate_errors`.
        """
        solutions, indicators = self.indicate_errors(**kwargs)

        features = {field: None for field in self.fields}
        targets = {field: None for field in self.fields}
        for subinterval in range(len(self)):
            for field in self.fields:
                features[field] = self.concat(
                    features[field],
                    self.collect_features(solutions, field, subinterval),
                )
                targets[field] = self.concat(
                    targets[field], self.collect_targets(indicators, field, subinterval)
                )
        return solutions, indicators, features, targets


class DataDrivenMeshSeq(GoalOrientedDataGenerator):
    """
    A :class:`MeshSeq` subclass, which enables the emulation of goal-oriented error
    estimtaion and mesh adaptation using neural networks.
    """

    def __init__(
        self,
        time_partition: TimePartition,
        initial_meshes: list,
        networks: dict,
        **kwargs
    ):
        super().__init__(time_partition, initial_meshes, **kwargs)
        self.networks = networks
        for network in self.networks.values():
            # TODO: Check of type torch.nn.Module
            network.eval()

    @PETSc.Log.EventDecorator()
    def generate_data(self, **kwargs) -> Tuple(AttrDict, np.array):
        """
        Generate data for running the data-driven error estimation method.

        Keyword arguments passed to :meth:`AdjointMeshSeq.solve_adjoint`.
        """
        solutions = self.solve_adjoint(**kwargs)

        features = {field: None for field in self.fields}
        for subinterval in range(len(self)):
            for field in self.fields:
                features[field] = self.concat(
                    features[field],
                    self.collect_features(solutions, field, subinterval),
                )
        return solutions, features

    @torch.no_grad
    def indicate_errors(self, **kwargs) -> Tuple(AttrDict, AttrDict):
        """
        Indicate errors using a data-driven approach on the provided neural network.

        Keyword arguments passed to :meth:`AdjointMeshSeq.solve_adjoint`.
        """
        solutions, features = self.generate_data(**kwargs)
        P0_spaces = [FunctionSpace(mesh, "DG", 0) for mesh in self]
        indicators = AttrDict(
            {[Function(P0) for P0 in P0_spaces] for field in self.fields}
        )
        for field, network in self.networks.items():
            for indicator, feature in zip(indicators[field], features[field]):
                indicator.dat.data[:] = network(feature)
        return solutions, indicators
