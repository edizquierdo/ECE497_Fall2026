##################################################################################
# Evolve the body AND the brain of a soft robot so that it walks to the right.
# One genotype holds both: first the body (5 genes per voxel), then the
# neural network (weights, biases, activation functions).
# Run this first; it saves the best genotype to best.npy (used by sim.py).
##################################################################################
import numpy as np
import fnn
import eas as ea
import env
from config import envname, duration, grid, period, layers, weightrange    # shared with sim.py (see config.py)

seed = None         # set to an integer to get the same run every time

if seed is not None:
    np.random.seed(seed)

# Parameters of the evolutionary algorithm
bodysize = grid*grid*env.materials
brainsize = fnn.FNN.genome_size(layers)
genesize = bodysize + brainsize
print("Number of parameters:",genesize,"(body:",bodysize,"+ brain:",brainsize,")")

popsize = 20
recombProb = 0.5
mutatStd = 0.1
generations = 30
demeSize = 5
eliteprop = 0.1

def fitnessFunction(genotype):
    # Step 1: Build the body from the first part of the genotype.
    body = env.makeBody(genotype[:bodysize], grid)
    if not env.isValid(body):
        return -1.0                 # a body that falls apart, or has no muscles, can't walk

    # Step 2: Create the neural network and set its parameters from the rest of the genotype.
    controller = fnn.FNN(layers)
    controller.weightrange = weightrange
    controller.biasrange = weightrange
    controller.setParams(genotype[bodysize:])

    # Step 3: Put the brain in the body, run the simulation, and add up how far it walked.
    robot = env.SoftRobot(body, controller, envname, duration, period)
    fitness = 0.0
    for t in range(duration):
        robot.think()
        reward, done = robot.move()
        fitness += reward
        if done:
            break
    robot.close()
    return fitness

# Evolve
ga = ea.Generational(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
ga.run()

avgfit, bestfit, bestind = ga.fitStats()
print("Best fitness (distance walked):",bestfit)
print("Best body (0 empty, 1 rigid, 2 soft, 3 horizontal muscle, 4 vertical muscle):")
print(env.makeBody(bestind[:bodysize], grid))
np.save("best.npy",bestind)
ga.showFitness()
