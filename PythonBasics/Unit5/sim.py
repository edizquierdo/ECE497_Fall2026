##################################################################################
# Simulate the best evolved soft robot (best.npy, from evolve.py), watch it walk
# in a window, and save its behavior to a file (used by viz.py).
##################################################################################
import numpy as np
import fnn
import env
from config import envname, duration, mode, body, grid, controller, loop, hidden, period, weightrange    # shared with evolve.py (see config.py)

render = True       # set to False to skip the animation window

def SimInd(genotype):
    # Build the body from the genotype (or use the fixed one), and the brain from the rest
    if mode == "codesign":
        bodysize = grid*grid*env.materials
        robotbody = env.makeBody(genotype[:bodysize], grid)
    else:
        bodysize = 0
        robotbody = env.loadBody(body)
    brain = fnn.FNN(env.brainLayers(controller, loop, robotbody.shape, hidden))
    brain.weightrange = weightrange
    brain.biasrange = weightrange
    brain.setParams(genotype[bodysize:])

    # Put the brain in the body
    robot = env.SoftRobot(robotbody, brain, controller, loop, envname, duration, period, render=render)

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
    return robotbody, pos, muscles

# Simulate and save data in a file
genotype = np.load('best.npy')
robotbody, pos, muscles = SimInd(genotype)
np.savez('sim.npz', body=robotbody, pos=pos, muscles=muscles)
