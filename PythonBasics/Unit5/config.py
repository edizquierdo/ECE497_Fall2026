##################################################################################
# Settings shared by evolve.py and sim.py, so the two always describe the same
# task, body grid, and network. Change them here, and both scripts follow.
##################################################################################

# Parameters of the task
envname = "Walker-v0"   # EvoGym task: walk as far right as possible on flat ground
duration = 300          # time steps per trial

# Parameters of the body
grid = 4                # the body is a grid x grid square of voxels

# Parameters of the neural network (the same network drives every muscle)
period = 25             # time steps per cycle of the clock fed into the network
layers = [4,4,1]        # inputs: clock (sin, cos) and the muscle's (x, y) position; output: the muscle's command
weightrange = 5         # genes are in [-1,1]; scale them so weights and biases are in [-5,5]
