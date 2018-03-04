import math

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Ackley(Function):

    def function(self, x):
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c ** 2.0
            secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(x))
        return -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e

    def getBounds(self):
        return Bounds(-32.768, 32.768)