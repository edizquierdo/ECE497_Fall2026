##################################################################################
# Example script for evolving a feedforward neural network to solve i/o task
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import fnn 
import eas as ea

# Parameters of the XOR task
dataset = [[-1,-1],[-1,1],[1,-1],[1,1]]
labels = [0,1,1,0]

# Parameters for another task
dataset = [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]]
labels = [1,1,1,1,1,1,1,1,0,0,0,0]

# Parameters of the neural network
layers = [2,5,1]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*5  # Which activation function of 5 possible 
print("Number of parameters:",genesize)

popsize = 100
recombProb = 0.5
mutatProb = 0.01
generations = 500
demeSize = 5
eliteprop = 0.1

def fitnessFunction(genotype):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype)
    
    # Step 3. For each training point in the dataset, evaluate the current neural network.
    error = 0.0
    for i in range(len(dataset)):
        temperror = np.abs(a.forward(dataset[i])[0,0] - labels[i])
        if temperror > 1:
            error += 1
        else:
            error += temperror
        #error += np.abs(np.clip(a.forward(dataset[i]),0,1) - labels[i])

    return 1 - (error/len(dataset))

# Evolve
ga = ea.Generational(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatProb=mutatProb, demeSize=demeSize, eliteprop=eliteprop)
ga.run()
ga.showFitness()      

# Obtain best final solution and create a neural network with it
avgfit, bestfit, bestind = ga.fitStats()
a = fnn.FNN(layers)
a.setParams(bestind)

# Function to visualize 
def viz(neuralnet, dataset, label):
    X = np.linspace(-1.05, 1.05, 100)
    Y = np.linspace(-1.05, 1.05, 100)
    output = np.zeros((100,100))
    i = 0
    for x in X: 
        j = 0
        for y in Y: 
            output[i,j] = neuralnet.forward([x,y])[0,0]
            j += 1
        i += 1
    plt.contourf(X,Y,output)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(dataset)):
        if label[i] == 1:
            plt.plot(dataset[i][0],dataset[i][1],'wo')
        else:
            plt.plot(dataset[i][0],dataset[i][1],'wx')
    plt.show()  

# Visualize data
viz(a, dataset, labels)
