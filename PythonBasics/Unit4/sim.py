##################################################################################
# Simulate the best evolved vehicle (best.npy, from evolve_fnn_embodiedtask.py)
# on many new trials and save its behavior to a file (used by viz.py).
##################################################################################
import numpy as np
import fnn
import env

# Parameters of the task (should match evolve_fnn_embodiedtask.py)
duration = 200
distance = 5
trials = 20         # number of random starting positions/headings to test

# Parameters of the neural network (must match evolve_fnn_embodiedtask.py)
layers = [2,4,2]
weightrange = 5

def SimInd(genotype):
    # Create the neural network and set the parameters according to the genotype
    controller = fnn.FNN(layers)
    controller.weightrange = weightrange
    controller.biasrange = weightrange
    controller.setParams(genotype)

    # Variables to store time-varying data for each trial
    xpos = np.zeros((trials,duration))
    ypos = np.zeros((trials,duration))
    dist = np.zeros((trials,duration))
    fitness = 0.0

    for i in range(trials):
        # First, instantiate a vehicle and a light source
        agent = env.Braitenberg(controller, distance)
        light = env.Light()

        # Run the simulation
        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()
            # Keep track of the data
            xpos[i][t] = agent.pos.x
            ypos[i][t] = agent.pos.y
            dist[i][t] = agent.distance(light)
            fitness += 1/(1 + dist[i][t])

    print("Fitness:",fitness/(trials*duration))
    return xpos,ypos,dist

# Simulate and save data in a file
genotype = np.load('best.npy')
a_x, a_y, d = SimInd(genotype)
np.savez('sim.npz', a_x=a_x, a_y=a_y, d=d)
