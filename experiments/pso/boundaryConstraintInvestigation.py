import matplotlib.pyplot as plt
import numpy as np

from experiments.pso.boundaryConstraintPSO import BoundaryConstraintPSO

ITERATIONS = 5000
ITERATIONS_SAMPLE_SIZE = 100
SAMPLES = 1

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
from experiments.problems.functions.quartic import Quartic
from experiments.problems.functions.salomon import Salomon
from experiments.problems.functions.schafferF6 import SchafferF6
from experiments.problems.functions.schwefelF2_26 import SchwefelF2_26

problem_list = [
    # AbsoluteValue(),
    # Ackley(),
    # Elliptic(),
    # Griewank(),
    # Hyperellipsoid(),
    # Quartic(),
    Salomon(),
    # SchafferF6(),
    # SchwefelF2_26(),
]

from experiments.pso.boundaryConstraints import *

boundaryConstraint_list = [
    (None, "None", 'k-'),
    (bc1, "Feasible position update", 'g+'),
    (bc2, "Clamping approach", 'b:'),
    (bc3, "Per element reinitialization", 'r--'),
    (bc4, "Per element reinitialization and velocity = 0", 'r+'),
    (bc5, "Initialize to personal best position", 'g:'),
    (bc6, "Initialize to personal best position and velocity = 0", 'b--'),
    (bc7, "Initialize to global best position", 'b+'),
    (bc8, "Initialize to global best position velocity = 0", 'g:'),
    (bc9, "Reverse velocity", 'o--'),
    (bc10, "Arithmetic average", 'g-'),
]

for problem in problem_list:

    bc_pso_errors_mean = []

    print("Problem: " + problem.getDescription())
    for boundaryConstraint in boundaryConstraint_list:

        print("Boundary Constraint: " + boundaryConstraint[1])
        pso_errors = []
        pso_error = []

        for i in range(SAMPLES):
            pso = BoundaryConstraintPSO()

            pso.boundaryConstraint = boundaryConstraint[0]

            pso.error = problem.function
            pso.bounds = problem.getBounds()
            pso.initialPosition = problem.getBounds()

            pso.num_particles = NUM_PARTICLES
            pso.num_dimensions = NUM_DIMENSIONS
            pso.weight = INERTIA_WEIGHT
            pso.cognitiveConstant = COGNITIVE_CONSTANT
            pso.socialConstant = SOCIAL_CONSTANT
            pso.vmax = V_MAX

            pso.color = 'black'
            trainingErrors, trainingError = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

            pso_errors.append(trainingErrors)
            pso_error.append(trainingError)
            print(".")

        iterations = [y[1] for y in pso_errors[0]]

        pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
        pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)

        pso_error_mean = np.mean(pso_error)
        pso_error_std = np.std(pso_error)

        bc_pso_errors_mean.append(pso_errors_mean)

        print(boundaryConstraint[1])
        print(pso_error_mean)
        print('(' + str(pso_error_std) + ')')

    plt.close()

    fig = plt.figure()
    plt.grid(1)
    plt.xlim([0, ITERATIONS])
    plt.ion()
    plt.xlabel('Iterations')
    plt.ylabel('Error')

    plots = []
    descriptions = []
    for i, boundaryConstraint in enumerate(boundaryConstraint_list):
        plots.append(plt.plot(iterations, bc_pso_errors_mean[i], boundaryConstraint[2], linewidth=1, markersize=3)[0])
        descriptions.append(boundaryConstraint[1])
    plt.legend(plots, descriptions)
    fig.savefig('Result' + problem.getDescription() + '.png')
    plt.show(5)