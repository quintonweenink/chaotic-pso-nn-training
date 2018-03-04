import math
import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Salomon(Function):

    def function(self, x):
        x_squared = np.power(x, 2)
        return -np.cos(2 * math.pi * np.sum(x_squared)) + (0.1 * np.sqrt(np.sum(x_squared + 1)))

    def getBounds(self):
        return Bounds(-1.28, 1.28)
