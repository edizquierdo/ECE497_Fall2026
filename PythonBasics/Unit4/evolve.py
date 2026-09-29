##################################################################################
# Evolve a feedforward neural network to control a Braitenberg vehicle so that
# it approaches a light source (phototaxis).
# Run this first; it saves the best genotype to best.npy (used by sim.py).
##################################################################################
import numpy as np
import fnn
import eas as ea
import env
from config import duration, distance, layers, weightrange    # shared with sim.py (see config.py)

# Parameters of the task
reps = 4            # trials (random starting position and heading) per evaluation
finaltrials = 100   # fresh trials used to re-evaluate the best individual at the end
seed = None         # set to an integer to get the same run every time

if seed is not None:
    np.random.seed(seed)

# Parameters of the evolutionary algorithm
genesize = fnn.FNN.genome_size(layers)
print("Number of parameters:",genesize)
print("Each fitness evaluation averages",reps,"random trials")

popsize = 50
recombProb = 0.5
mutatStd = 0.05
generations = 100
demeSize = 5
eliteprop = 0.1

def fitnessFunction(genotype, reps=reps):
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
print("Best fitness (recorded during evolution):",bestfit)

# The recorded fitness comes from only a few random trials, so check it on many new ones.
print("Same individual on",finaltrials,"fresh trials:",fitnessFunction(bestind, reps=finaltrials))
np.save("best.npy",bestind)
ga.showFitness()
