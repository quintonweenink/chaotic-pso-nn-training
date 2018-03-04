import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class AbsoluteValue(Function):

    def function(self, position):
        return np.sum(np.abs(position))

    def getBounds(self):
        return Bounds(-100, 100)

    def test(self):
        assert(3 == self.function(np.array([1, 2])))
        assert(6 == self.function(np.array([-2, 4])))