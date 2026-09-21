##################################################################################
# Feed-forward neural network for teaching and learning.
# Works with any number of hidden layers/neurons.
# Supports the following activation functions: sigmoidm, tanh, relu, linear, gaussian, and identity.
##################################################################################

import numpy as np

# ----------------------------------------------
# Feedforward Artificial Neural Network (v1)
# Input layer, hidden layer, and output layer
# Implemented using for loops for pedagological reasons
# ----------------------------------------------
class ANNv1:
    
    def __init__(self,NI,NH,NO):
        self.nI = NI
        self.nH = NH
        self.nO = NO
        # Parameters of this NN
        self.wIH = np.random.random(size=(NI,NH))
        self.wHO = np.random.random(size=(NH,NO))
        self.bH = np.random.random(size=NH)
        self.bO = np.random.random(size=NO) 
        # State of NN
        self.I = np.zeros(NI)
        self.H = np.zeros(NH)
        self.O = np.zeros(NO)
        
    def calc(self, inputvector):
        self.I = inputvector
        for h in range(self.nH):
            netinput = self.bH[h]
            for i in range(self.nI):
                netinput += self.I[i] * self.wIH[i,h]
            self.H[h] = sigmoid(netinput)
        for o in range(self.nO):
            netinput = self.bO[o]            
            for h in range(self.nH):
                netinput += self.H[h] * self.wHO[h,o]
            self.O[o] = sigmoid(netinput)
        return self.O


# ----------------------------------------------
# Feedforward Artificial Neural Network (v2)
# Same architecture as before: Input layer, hidden layer, and output layer
# This time implemented more efficiently using dot products 
# ----------------------------------------------     
class ANNv2:

    def __init__(self, NIU, NHU, NOU):
        self.nI = NIU
        self.nH = NHU
        self.nO = NOU
        self.wIH = np.random.normal(0,5,size=(NIU,NHU)) #np.zeros((NIU,NHU))
        self.wHO = np.random.normal(0,5,size=(NHU,NOU)) #np.zeros((NHU,NOU))
        self.bH = np.random.normal(0,5,size=NHU) #np.zeros(NHU)
        self.bO = np.random.normal(0,5,size=NOU) #np.zeros(NOU)
        self.HiddenActivation = np.zeros(NHU)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

    def step(self,Input):
        self.Input = np.array(Input)
        self.HiddenActivation = sigmoid(np.dot(self.Input.T,self.wIH)+self.bH)
        self.OutputActivation = sigmoid(np.dot(self.HiddenActivation,self.wHO)+self.bO)
        return self.OutputActivation

    def output(self):
        return self.OutputActivation
    

# ----------------------------------------------
# Feedforward Artificial Neural Network (v3)
# Same architecture as before: Input layer, hidden layer, and output layer
# This time implemented more efficiently using dot products 
# ---------------------------------------------- 
class FNN:
    def __init__(self, units_per_layer):
        """ Create Feedforward Neural Network based on specifications
        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)

        # lambdas for supported activation functions
        self.activation_funcs = {
            0: lambda x: 1 / (1 + np.exp(-x)),              # sigmoid
            1: lambda x: 2 / (1 + np.exp(-2 * x)) - 1,      # tanh
            2: lambda x: np.maximum(0, x),                  # relu
            3: lambda x: x,                                 # linear
            4: lambda x: np.exp(-x**2)                      # gaussian
        }

        self.weightrange = 10
        self.biasrange = 10

    def setParams(self, params):
        """ Set the weights, biases, and activation functions of the neural network 
        Weights and biases are set directly by a parameter;
        The activation function for each layer is set by the parameter with the highest value (one for each possible one out of the six)
        """
        self.weights = []
        start = 0
        # Weights 
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            self.weights.append((params[start:end]*self.weightrange).reshape(self.units_per_layer[l],self.units_per_layer[l+1]))
            start = end
        # Biases             
        self.biases = []
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            self.biases.append((params[start:end]*self.biasrange).reshape(1,self.units_per_layer[l+1]))
            start = end
        # Transfer functions 
        self.activation = []
        for l in np.arange(self.num_layers-1):
            end = start + len(self.activation_funcs)
            actfunc = np.argmax(params[start:end])
            self.activation.append(self.activation_funcs[actfunc])            
            start = end

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network """
        states = np.asarray(inputs)
        for l in np.arange(self.num_layers - 1):
            if states.ndim == 1:
                states = [states]
            states = self.activation[l](np.matmul(states, self.weights[l]) + self.biases[l])
        return states

