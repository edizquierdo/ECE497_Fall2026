import numpy as np
import perceptron as pt
import matplotlib.pyplot as plt

dur = 20000 # Number of learning trials

dataset = [[-1,-1],[-1,1],[1,-1],[1,1]] # Same as before, but instead of 0s and 1s,... -1s and 1s.. 
target = [0,1,1,0]  # So this would mean we are training for XOR !!! 

def train(dur, dataset, target):
    a = pt.NeuralNet(2,2) # Initialize a neural network with 2 inputs and 2 hidden units, we assume one output for now.
    error = np.zeros(dur) # An array to keep track of the learning that happens over time
    for i in range(dur):
        error[i] = 0.0  # We init our error to 0, then accumulate the error for each data point in our training dataset
        for j in range(len(dataset)):
            error[i] += a.train(dataset[j],target[j])

    # plt.plot(error)
    # plt.xlabel("Training time")
    # plt.ylabel("Error")
    # plt.show()        
    # a.viz(dataset,target,100)
    return error

repetitions = 100
error = np.zeros((repetitions, dur))
for r in range(repetitions):
    error[r] = train(dur, dataset, target)
    
# Viz
plt.plot(error.T)
plt.xlabel("Training time")
plt.ylabel("Error")
plt.show()

