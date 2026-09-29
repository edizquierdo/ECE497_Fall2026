##################################################################################
# Settings shared by evolve.py and sim.py, so the two always describe the same
# task and the same network. Change them here, and both scripts follow.
##################################################################################

# Parameters of the task
duration = 200      # time steps per trial
distance = 5        # starting distance from the light

# Parameters of the neural network
layers = [2,4,2]
weightrange = 5     # genes are in [-1,1]; scale them so weights and biases are in [-5,5]
