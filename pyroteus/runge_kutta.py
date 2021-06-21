"""
Runge-Kutta timestepping for Pyroteus.

Code largely copied from [the Thetis project](https://thetisproject.org).
"""
from abc import ABCMeta, abstractproperty
import numpy as np


class ButcherTableau(object):
    """
    Abstract class defining the Butcher tableaux
    associated with Runge-Kutta schemes.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def a(self):
        """
        Runge-Kutta matrix
        """
        pass

    @abstractproperty
    def b(self):
        """
        Weights of Butcher tableau
        """
        pass

    @abstractproperty
    def c(self):
        """
        Nodes of Butcher tableau
        """
        pass

    def __init__(self):
        super(ButcherTableau, self).__init__()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), "Butcher tableau must be lower diagonal"
        assert np.allclose(np.sum(self.a, axis=1), self.c), "Inconsistent Butcher tableau"

        self.num_stages = len(self.b)
        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()


class ForwardEuler(ButcherTableau):
    a = [[0.0]]
    b = [1.0]
    c = [0.0]


class BackwardEuler(ButcherTableau):
    a = [[1.0]]
    b = [1.0]
    c = [1.0]


class CrankNicolson(ButcherTableau):
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]


class SSPRK33(ButcherTableau):
    a = [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.25, 0.25, 0.0]]
    b = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    c = [0.0, 1.0, 0.5]
