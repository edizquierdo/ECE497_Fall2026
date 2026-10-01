##################################################################################
# Simulate the best evolved soft robot (best.npy, from evolve.py), watch it walk
# in a window, and save its behavior to a file (used by viz.py).
##################################################################################
import numpy as np
import fnn
import env
from config import envname, duration, grid, period, layers, weightrange    # shared with evolve.py (see config.py)

render = True       # set to False to skip the animation window

def SimInd(genotype):
    # Build the body and the neural network from the genotype
    bodysize = grid*grid*env.materials
    body = env.makeBody(genotype[:bodysize], grid)
    controller = fnn.FNN(layers)
    controller.weightrange = weightrange
    controller.biasrange = weightrange
    controller.setParams(genotype[bodysize:])

    # Put the brain in the body
    robot = env.SoftRobot(body, controller, envname, duration, period, render=render)

    # Variables to store time-varying data
    pos = np.zeros((duration,2))
    muscles = np.zeros((duration,len(robot.muscles)))
    fitness = 0.0

    # Run the simulation
    for t in range(duration):
        robot.think()
        reward, done = robot.move()
        # Keep track of the data
        pos[t] = robot.position()
        muscles[t] = robot.muscles
        fitness += reward
        if done:
            pos, muscles = pos[:t+1], muscles[:t+1]
            break
    robot.close()

    print("Fitness (distance walked):",fitness)
    return body, pos, muscles

# Simulate and save data in a file
genotype = np.load('best.npy')
body, pos, muscles = SimInd(genotype)
np.savez('sim.npz', body=body, pos=pos, muscles=muscles)
