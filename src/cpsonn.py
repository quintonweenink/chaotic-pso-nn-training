import numpy as np

from src.chaoticParticle import ChaoticParticle
from src.numberGenerator.chaos.cprng import CPRNG

from mlpy.psoNeuralNetwork.psonn import PSONN

class CPSONN(PSONN):
    def __init__(self):
        super(CPSONN, self).__init__()

        self.numberGenerator = None

    def createParticles(self):
        assert isinstance(self.numberGenerator, CPRNG) == True

        for i in range(self.pso.num_particles):
            self.pso.swarm.append(ChaoticParticle(self.bounds, self.numberGenerator, self.inertia_weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.pso.swarm[i].initPos(position, velocity)

