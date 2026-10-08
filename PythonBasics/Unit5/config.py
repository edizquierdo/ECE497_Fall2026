##################################################################################
# Settings shared by evolve.py and sim.py, so the two always describe the same
# task, body, and brain. Change them here, and both scripts follow.
##################################################################################

# Parameters of the task
envname = "Walker-v0"   # EvoGym task: walk as far right as possible on flat ground
duration = 300          # time steps per trial

# What to evolve
mode = "codesign"       # "codesign": evolve the body AND the brain together
                        # "control":  keep the body fixed (below) and evolve only the brain

# Parameters of the body
body = "biped"          # [control] the fixed body: a preset ("biped", "worm", "block", "tripod")
                        # or a text file you designed, like "my_body.txt"
grid = 5                # [codesign] the body is evolved on a grid x grid square of voxels

# Parameters of the brain (see env.py for what each one sees)
controller = "local"    # "local": one small network, copied into every muscle
                        # "global": one network for the whole body, one output per grid cell
loop = "closed"         # "open": the brain only sees a clock (and, if local, the muscle's position)
                        # "closed": it also senses how stretched the voxels are
hidden = 8 if controller == "local" else 16     # neurons in the hidden layer
period = 25             # time steps per cycle of the clock fed into the brain
weightrange = 5         # genes are in [-1,1]; scale them so weights and biases are in [-5,5]
