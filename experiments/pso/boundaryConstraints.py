import numpy as np

def outsideSearchSpace(position, bounds):
    greaterThanMaxBound = np.greater(position, bounds.maxBound)
    lessThanMinBound = np.less(position, bounds.minBound)
    return np.logical_or(lessThanMinBound, greaterThanMaxBound)

"""
Update the personal best positions only if the new particle position is better than its current personal
best position, and if the new particle position is feasible. That is, a new particle position can not become
a personal best position if it violates boundary constraints.
"""
def bc1(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        if not np.any(outsideSearchSpace(pso.swarm[j].position, pso.bounds)):
            pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

"""
Clamping approach: If a particle violates a boundary constraint in a specific dimension, then clamp the
corresponding decision variable at the boundary value. That is, if x ij (t + 1) < x min,j then x ij (t + 1) =
x min,j , or if x ij (t + 1) > x max,j then x ij (t + 1) = x max,j .
"""
def bc2(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        pso.swarm[j].position = np.clip(pso.swarm[j].position, pso.bounds.minBound, pso.bounds.maxBound)

"""
Per element reinitialization: For any decision variable of any particle that violates a boundary constraint,
reinitialize that decision variable to a random position that satisfies the boundary constraints. That is, if
x ij (t + 1) < x min,j or x ij (t + 1) > x max,j , then x ij (t + 1) ∼ U (x min,j , x max,j ).
"""
def bc3(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        uniformList = (pso.initialPosition.maxBound - pso.initialPosition.minBound) * \
                      np.random.random(np.sum(outList)) + pso.initialPosition.minBound
        pso.swarm[j].position[outList] = uniformList


"""
Per element reinitialization and setting velocity to zero: Adapt the per element reinitialization approach
above to also set the velocity of the decision variable that violates a boundary constraint to zero. The
corresponding decision variable’s new position will therefore not be influenced by the momentum term.
"""
def bc4(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        uniformList = (pso.initialPosition.maxBound - pso.initialPosition.minBound) * \
                      np.random.random(np.sum(outList)) + pso.initialPosition.minBound
        pso.swarm[j].position[outList] = uniformList
        pso.swarm[j].velocity[outList] = np.zeros(np.sum(outList))

"""
Initialize to personal best position: Initialize the boundary violating decision variable to the corresponding
personal best position. That is, if x ij (t + 1) violates a baoundary constraint, then x ij (t + 1) = y ij (t).
"""
def bc5(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        pso.swarm[j].position[outList] = pso.swarm[j].best_position[outList]

"""
Initialize to personal best position and set velocity to zero: Adapt the intialize to personal best position
strategy above to also set the corresponding velocity to 0.
"""
def bc6(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        pso.swarm[j].position[outList] = pso.swarm[j].best_position[outList]
        pso.swarm[j].velocity[outList] = np.zeros(np.sum(outList))

"""
Initialize to global best position: As for the above, but x ij (t + 1) = ŷ j (t) for the boundary violating
decision variable.
"""
def bc7(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        pso.swarm[j].position[outList] = pso.group_best_position[outList]

"""
Initialize to global best position and set velocity to zero: Adapt the intialize to global best position
strategy to also set the corresponding velocity to 0.
"""
def bc8(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        pso.swarm[j].position[outList] = pso.group_best_position[outList]
        pso.swarm[j].velocity[outList] = np.zeros(np.sum(outList))

"""
Reverse velocity: The velocity of the bounadry violating decision variable is simply reversed while that
decision variable violates the boundary constraint.
"""
def bc9(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        firstIterationOutside = np.logical_and(outList, np.logical_not(pso.swarm[j].violatedBoundaryConstraint))
        pso.swarm[j].velocity[firstIterationOutside] = -pso.swarm[j].velocity[firstIterationOutside]
        pso.swarm[j].violatedBoundaryConstraint = outList

"""
Set the bounadry violating decision variable to an arithmetic average of the corresponding personal best
and global best position. That is, x ij (t + 1) = αy ij (t) + (1 − α)ŷ j (t), where α ∼ U (0, 1) (randomly selected
in the range (0, 1)).
"""
def bc10(pso):
    for j in range(pso.num_particles):
        pso.swarm[j].error = pso.error(pso.swarm[j].position)

        pso.swarm[j].getPersonalBest()

    pso.getGlobalBest()

    for j in range(pso.num_particles):
        pso.swarm[j].update_velocity(pso.group_best_position)
        pso.swarm[j].update_position()

        outList = outsideSearchSpace(pso.swarm[j].position, pso.bounds)
        a = np.random.random(np.sum(outList))
        pso.swarm[j].position[outList] = (a * pso.swarm[j].position[outList]) + ((1-a) * pso.group_best_position[outList])