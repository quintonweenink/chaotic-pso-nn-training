import matplotlib.pyplot as plt
import numpy as np

from mlpy.particleSwarmOptimization.pso import PSO

np.set_printoptions(suppress=True)

class BoundaryConstraintPSO(PSO):

    def __init__(self):
        super(BoundaryConstraintPSO, self).__init__()

        self.boundaryConstraint = None

    def loopOverParticles(self):

        for j in range(len(self.swarm)):
            self.swarm[j].violatedBoundaryConstraint = np.zeros(self.num_dimensions, dtype=bool)
        if self.boundaryConstraint != None:
            self.boundaryConstraint(self)
        else:
            super(BoundaryConstraintPSO, self).loopOverParticles()