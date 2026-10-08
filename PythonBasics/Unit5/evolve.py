##################################################################################
# Evolve a soft robot that walks to the right, in one of two modes (see config.py):
#   - "codesign": evolve the body AND the brain together. One genotype holds
#     both: first the body (5 genes per voxel), then the neural network
#     (weights, biases, activation functions).
#   - "control": keep the body fixed and evolve only the brain.
# Run this first; it saves the best genotype to best.npy (used by sim.py).
##################################################################################
import numpy as np
import fnn
import eas as ea
import env
from config import envname, duration, mode, body, grid, controller, loop, hidden, period, weightrange    # shared with sim.py (see config.py)

seed = None         # set to an integer to get the same run every time

if seed is not None:
    np.random.seed(seed)

# The body's genes (none, if the body is fixed) and the brain's layers
if mode == "codesign":
    bodysize = grid*grid*env.materials
    shape = (grid, grid)
else:
    bodysize = 0
    fixedbody = env.loadBody(body)
    shape = fixedbody.shape
    print("Fixed body (0 empty, 1 rigid, 2 soft, 3 horizontal muscle, 4 vertical muscle):")
    print(fixedbody)
layers = env.brainLayers(controller, loop, shape, hidden)

# Parameters of the evolutionary algorithm
brainsize = fnn.FNN.genome_size(layers)
genesize = bodysize + brainsize
print("Brain:",loop+"-loop",controller,"network with layers",layers)
print("Number of parameters:",genesize,"(body:",bodysize,"+ brain:",brainsize,")")

popsize = 50
recombProb = 0.5
mutatStd = 0.1
generations = 50
demeSize = 5
eliteprop = 0.1

def fitnessFunction(genotype):
    # Step 1: Build the body from the first part of the genotype (or use the fixed one).
    if mode == "codesign":
        robotbody = env.makeBody(genotype[:bodysize], grid)
        if not env.isValid(robotbody):
            return -1.0             # a body that falls apart, or has no muscles, can't walk
    else:
        robotbody = fixedbody

    # Step 2: Create the neural network and set its parameters from the rest of the genotype.
    brain = fnn.FNN(layers)
    brain.weightrange = weightrange
    brain.biasrange = weightrange
    brain.setParams(genotype[bodysize:])

    # Step 3: Put the brain in the body, run the simulation, and add up how far it walked.
    robot = env.SoftRobot(robotbody, brain, controller, loop, envname, duration, period)
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
ga = ea.Microbial(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
ga.run()

avgfit, bestfit, bestind = ga.fitStats()
print("Best fitness (distance walked):",bestfit)
if mode == "codesign":
    print("Best body (0 empty, 1 rigid, 2 soft, 3 horizontal muscle, 4 vertical muscle):")
    print(env.makeBody(bestind[:bodysize], grid))
np.save("best.npy",bestind)
ga.showFitness()
