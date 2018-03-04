import math

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Griewank(Function):

    def function(self, x):
        part1 = 0
        part2 = None
        for i in range(len(x)):
            part1 += x[i] ** 2
            part2 = 1
        for i in range(len(x)):
            part2 *= math.cos(float(x[i]) / math.sqrt(i + 1))
        return 1 + (float(part1) / 4000.0) - float(part2)

    def getBounds(self):
        return Bounds(-600, 600)