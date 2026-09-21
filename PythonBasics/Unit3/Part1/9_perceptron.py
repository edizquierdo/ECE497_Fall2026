import numpy as np

def step(x):
    return x>0

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Perceptron():

    def __init__(self, inputs):
        self.W = np.random.random(size=(inputs))*2 - 1
        self.bias = np.random.random()*2 - 1
        self.lc = 0.1 
    
    def forward(self, I):
        y = np.dot(I,self.W) + self.bias 
        return sigmoid(y)
