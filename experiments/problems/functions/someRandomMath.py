import math

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class SomeRandomMath(Function):

    def function(self, x):
        err = 0.0
        for i in range(len(x)):
            xi = x[i]
            err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
        return err

    def getBounds(self):
        return Bounds(-10, 10)