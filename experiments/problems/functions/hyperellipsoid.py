import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Hyperellipsoid(Function):

    def function(self, x):
        return np.sum(np.arange(1, len(x) + 1) * np.power(x, 2))

    def getBounds(self):
        return Bounds(-5.12, 5.12)

    def test(self):
        assert(3 == self.function(np.array([1, 1])))
        assert(12 == self.function(np.array([2, 2])))