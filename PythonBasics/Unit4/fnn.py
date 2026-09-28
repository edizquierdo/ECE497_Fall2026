##################################################################################
# Feed-forward neural network for teaching and learning.
# Works with any number of hidden layers/neurons.
# Supports the following activation functions: sigmoid, tanh, relu, linear, and gaussian.
##################################################################################
import numpy as np

# Large weights can push sigmoid/tanh inputs high enough to overflow exp();
# the result (0.0) is still correct, so silence the warning.
np.seterr(over='ignore')

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

        # Initialization of weights and biases
        self.weightrange = 0.1
        self.biasrange = 0.1

    def setParams(self, params):
        """ Set the weights, biases, and activation functions of the neural network 
        Weights and biases are set directly by a parameter;
        The activation function for each layer is set by the parameter with the highest value (one for each possible one out of the five)
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
            actfunc = 4 # FORCED TO BE SIGMOID np.argmax(params[start:end])
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

