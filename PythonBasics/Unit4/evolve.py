##################################################################################
# Evolve a feedforward neural network to control a Braitenberg vehicle so that
# it approaches a light source (phototaxis).
# Run this first; it saves the best genotype to best.npy (used by sim.py).
##################################################################################
import numpy as np
import fnn
import eas as ea
import env

# Parameters of the task
duration = 200      # time steps per trial
distance = 5        # starting distance from the light
reps = 4            # trials (random starting position and heading) per evaluation

# Parameters of the neural network
layers = [2,4,2]
weightrange = 5     # genes are in [-1,1]; scale them so weights and biases are in [-5,5]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*5  # Which activation function of 5 possible
print("Number of parameters:",genesize)

popsize = 50
recombProb = 0.5
mutatStd = 0.05
generations = 50
demeSize = 5
eliteprop = 0.1

def fitnessFunction(genotype):
    # Step 1: Create the neural network and set the parameters according to the genotype.
    controller = fnn.FNN(layers)
    controller.weightrange = weightrange
    controller.biasrange = weightrange
    controller.setParams(genotype)

    fitness = 0.0
    for r in range(reps):
        # Step 2: Create the body and the environment.
        agent = env.Braitenberg(controller, distance)
        light = env.Light()

        # Step 3: Run the simulation, rewarding the agent for being close to the light at every step.
        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()
            fitness += 1/(1 + agent.distance(light))    # 1 on top of the light, towards 0 far away

    return fitness/(reps*duration)

# Evolve
ga = ea.Microbial(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
ga.run()
avgfit, bestfit, bestind = ga.fitStats()
print("Best fitness:",bestfit)
np.save("best.npy",bestind)
ga.showFitness()
