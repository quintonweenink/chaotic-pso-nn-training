import math

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class SchafferF6(Function):

    def function(self, x):
        if len(x) <= 2:
            return self.schafferF6HelperFunction(0, x)
        else:
            total = 0
            for i in range(len(x)):
                total += self.weights[i] * \
                         self.schafferF6HelperFunction(i, x)
            return total

    def schafferF6HelperFunction(self, i, x):
        xi = x[i]
        yi = x[(i + 1) % len(x)]
        sqr = xi ** 2 + yi ** 2
        return 0.5 + (math.sin(math.sqrt(sqr)) - 0.5) / (1 + 0.001 * sqr) ** 2

    def getBounds(self):
        return Bounds(-1.28, 1.28)