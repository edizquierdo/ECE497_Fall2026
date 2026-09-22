import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return x>=0

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Perceptron():

    def __init__(self, inputs):
        self.W = 1*(np.random.random(size=(inputs))*2 - 1)
        self.bias = 1*(np.random.random()*2 - 1)
        self.lc = 0.1 
    
    def forward(self, I):
        y = np.dot(I,self.W) + self.bias 
        return sigmoid(y)

class NeuralNet():

    def __init__(self, inputUnits, hiddenUnits):  # We will make an NN with x inputs and y hidden units, but only one output for now
        self.ni = inputUnits # number of inputs to the neural network
        self.nh = hiddenUnits
        # This is our set of Hidden units, they receive input from the input units
        self.H = [Perceptron(inputUnits) for i in range(self.nh)] # This is a shortcut for writing a for loop 
        # We also need to establish where we are going to save the outputs of those hidden units 
        self.hiddenOutput = np.zeros(hiddenUnits)
        # This is our Outout unit
        self.nO = Perceptron(hiddenUnits)
        # We also establish the place to store its output
        self.output = 0 
        self.lc = 0.1  # learning rate 

    def forward(self, Input): 
        for i in range(self.nh):
            # To calculate the output of each neuron in the hidden unit, 
            # We iterate througgh the number of hidden units
            # And for each one, we simply ask it to pass forward the Input vector that has been provided
            self.hiddenOutput[i] = self.H[i].forward(Input)
        # After that, we have the outputs of the hidden units, we pass that to the output neuron
        self.output = self.nO.forward(self.hiddenOutput)
        return self.output 
    
    def train(self, Input, target):
        #1. Propagate the input forward through the network to figure out the current output for that data point
        self.forward(Input)

        #2. Calculate error for output neuron
        derivative = self.output * (1 - self.output) # this is derivative of sigmoid
        errorOutput = (target - self.output) * derivative 

        #3. Calculate error for hidden units (backpropagate the error)
        error = np.zeros(self.nh) # each neuron in the hidden unit will have its own error (responsability for the error)

        for i in range(self.nh):
            derivative = self.hiddenOutput[i] * (1 - self.hiddenOutput[i]) # same as before
            error[i] = self.nO.W[i] * errorOutput * derivative 
            # Here, we multiply the weight between that neuron and the output neuron, the error, and derivative (see slides)
        
        #4. Update the weights, start with the hidden-output weights 
        # There's only one output neuron, we update ALL its incoming weights and its bias
        for i in range(self.nh):
            self.nO.W[i] += self.hiddenOutput[i] * errorOutput * self.lc 
        self.nO.bias += 1 * errorOutput * self.lc

        #5. Update all the input - to - hidden weights 
        # We do the same thing we did for the output neuron
        # This time to EACH of the neurons in the hidden layer 
        for i in range(self.nh):
            for j in range(self.ni):
                self.H[i].W[j] += Input[j] * error[i] * self.lc 
            self.H[i].bias += 1 * error[i] * self.lc 

        # We are done, let's return the error, same as before
        return abs(errorOutput)
    
    def viz(self,dataset,target,density,ax=None):
        # Density will be the number of points per dimension we will be testing!
        # Let's create a lot of X and Y points in the possible space of input data
        if ax is None:
            ax = plt.gca() # Fall back to the current axes if none was given
        X = np.linspace(-1.05, 1.05, density)
        Y = np.linspace(-1.05, 1.05, density)
        output = np.zeros((density,density))
        i = 0
        for x in X:
            j = 0
            for y in Y:
                output[i,j] = self.forward([x,y]) # We provide the network with the test input, and record the output
                j += 1
            i += 1
        ax.contourf(X,Y,output)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        for i in range(len(dataset)): # Let's also plot the data we used for training as disks
            if target[i] == 1:
                ax.plot(dataset[i][0],dataset[i][1],'wo')
            else:
                ax.plot(dataset[i][0],dataset[i][1],'wx')