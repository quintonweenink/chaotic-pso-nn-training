import math
import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Elliptic(Function):

    def function(self, x):
        return np.sum(np.power(np.power(10., 6 ), np.divide(np.arange(len(x)), np.subtract(x, 1.))))

    def getBounds(self):
        return Bounds(-100, 100)

    def test(self):
        assert(1 == self.function(np.array([5])))
        assert(math.pow(10, 6) + 1 == self.function(np.array([5, 2])))