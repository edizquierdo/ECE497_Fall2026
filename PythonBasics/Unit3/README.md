# Unit 3: Training Neural Networks

Two ways to train a simple feedforward neural network, both implemented from scratch in plain Python/NumPy so you can see every step.

## Part 1: Backpropagation

A minimal `Perceptron`/`NeuralNet` implementation (`perceptron.py`) trained with the backpropagation algorithm on the XOR problem (`train.py`). Shows how errors are computed at the output, propagated backward to the hidden units, and used to update weights step by step.

## Part 2: Evolutionary Algorithms

A more general feedforward network (`fnn.py`) whose weights, biases, and activation functions are encoded as a single genotype, trained by evolving a population of candidate solutions instead of computing gradients (`eas.py`). Includes several evolutionary strategies (generational GA, microbial GA, hill climbers) that can be swapped in and compared (`evolve.py`, `compare.py`).
