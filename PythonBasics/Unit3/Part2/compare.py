
import numpy as np
import matplotlib.pyplot as plt
import fnn 
import eas as ea

# Parameters of the XOR task
# dataset = [[-1,-1],[-1,1],[1,-1],[1,1]]
# labels = [0,1,1,0]

# Parameters for another task
dataset = [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]]
labels = [1,1,1,1,1,1,1,1,0,0,0,0]

popsize = 100
recombProb = 0.5
mutatStd = 0.01
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

# Running several repetitions
reps = 100
# Parameters of the neural network
layers = [2,2,1]
# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*5  # Which activation function of 5 possible 
print("Number of parameters:",genesize)
for r in range(reps):
    ga = ea.Generational(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
    ga.run()
    plt.plot(ga.bestHistory,'r')
plt.xlabel("Generations")
plt.ylabel("Best fitness")
plt.title("Best fitness over generations (5 runs)")

# Running several repetitions
# Parameters of the neural network
layers = [2,5,1]
# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*5  # Which activation function of 5 possible 
print("Number of parameters:",genesize)
for r in range(reps):
    ga = ea.Generational(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
    ga.run()
    plt.plot(ga.bestHistory,'b')
plt.xlabel("Generations")
plt.ylabel("Best fitness")
plt.title("Best fitness over generations (5 runs)")

# Running several repetitions
# Parameters of the neural network
layers = [2,2,2,1]
# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*5  # Which activation function of 5 possible 
print("Number of parameters:",genesize)
for r in range(reps):
    ga = ea.Generational(fitnessFunction, genesize, generations, popsize=popsize, recombProb=recombProb, mutatStd=mutatStd, demeSize=demeSize, eliteprop=eliteprop)
    ga.run()
    plt.plot(ga.bestHistory,'g')
plt.xlabel("Generations")
plt.ylabel("Best fitness")
plt.title("Best fitness over generations (5 runs)")
plt.show()
