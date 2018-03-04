import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Quartic(Function):

    def function(self, x):
        return np.sum(np.multiply(np.arange(1, len(x) + 1), np.power(x, 4.0)))

    def getBounds(self):
        return Bounds(-1.28, 1.28)

    def test(self):
        assert(16 == self.function(np.array([2])))
        assert(48 == self.function(np.array([2, 2])))