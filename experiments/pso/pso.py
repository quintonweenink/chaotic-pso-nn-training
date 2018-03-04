import matplotlib.pyplot as plt
import numpy as np

from mlpy.particleSwarmOptimization.pso import PSO

# Error arrays
pso_errors = []
pso_error = []
pso_training_error = []

ITERATIONS = 2000
ITERATIONS_SAMPLE_SIZE = 100
SAMPLES = 2

NUM_PARTICLES = 30
NUM_DIMENSIONS = 30
INERTIA_WEIGHT = 0.7
COGNITIVE_CONSTANT = SOCIAL_CONSTANT = 1.4

V_MAX = 0.1

from experiments.problems.functions.absoluteValue import AbsoluteValue
from experiments.problems.functions.ackley import Ackley
from experiments.problems.functions.elliptic import Elliptic
from experiments.problems.functions.griewank import Griewank
from experiments.problems.functions.hyperellipsoid import Hyperellipsoid

problem_list = [
    AbsoluteValue(),
    Ackley(),
    Elliptic(),
    Griewank(),
    Hyperellipsoid(),
]

boundaryConstraint_list = [
    None,
]

for problem in problem_list:
    print("Problem: " + problem.getDescription())
    for boundaryConstraint in boundaryConstraint_list:
        for i in range(SAMPLES):
            print("Iteration: " + str(i))
            pso = PSO()
            pso.bounds = problem.getBounds()
            pso.initialPosition = problem.getBounds()
            pso.num_particles = NUM_PARTICLES
            pso.num_dimensions = NUM_DIMENSIONS
            pso.weight = INERTIA_WEIGHT
            pso.cognitiveConstant = COGNITIVE_CONSTANT
            pso.socialConstant = SOCIAL_CONSTANT
            pso.vmax = V_MAX

            pso.error = problem.function

            pso.color = 'black'
            trainingErrors, trainingError = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

            pso_errors.append(trainingErrors)
            pso_error.append(trainingError)
            pso_training_error.append(trainingError)

    iterations = [y[1] for y in pso_errors[0]]

    # Random
    pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
    pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)
    print(pso_errors_mean)

    pso_error_mean = np.mean(pso_error)
    pso_error_std = np.std(pso_error)

    pso_training_error_mean = np.mean(pso_training_error)
    pso_training_error_std = np.std(pso_training_error)

    pso_training_factor = np.divide(pso_training_error, pso_error)
    pso_training_factor_mean = np.mean(pso_training_factor)
    pso_training_factor_std = np.std(pso_training_factor)

    print('- Random:')
    print(pso_error_mean)
    print('(' + str(pso_error_std) + ')')
    print(pso_training_error_mean)
    print('(' + str(pso_training_error_std) + ')')

    plt.close()

    print(iterations)
    print(pso_errors_mean[0])

    fig = plt.figure()
    plt.grid(1)
    plt.xlim([0, ITERATIONS])
    plt.ion()
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    random, = plt.plot(iterations, pso_errors_mean, 'k-', linewidth=1, markersize=3)
    plt.legend([random], ['Uniform'])
    fig.savefig('Result' + 'Vmax' + str(V_MAX) + '.png')
    plt.show(5)