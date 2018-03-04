import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class SchwefelF2_26(Function):

    def function(self, x):
        return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def getBounds(self):
        return Bounds(-1.28, 1.28)

    def test(self):
        assert (-np.sin(1) == self.function(np.array([1])))