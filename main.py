import numpy as np

class ANN:
    def __init__(self, sizes):
        # Initialize the weights and biases for the layers
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.normal(0.0, sizes[i+1]**-0.5, (sizes[i+1], sizes[i])))
            self.biases.append(np.zeros(sizes[i+1]))
        
    def sigmoid(self, x):
        # Implement the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, inputs):
        # Perform a forward pass through the network
        activations = inputs
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations) + self.biases[i]
            activations = self.sigmoid(z)
        return activations
    
    def train(self, inputs, targets, learning_rate):
        # Train the network
        # First, perform a forward pass to get the outputs
        activations = self.forward_pass(inputs)
        
        # Calculate the error
        errors = targets - activations
        
        # Calculate the gradients of the error with respect to the weights and biases
        grad_weights = []
        grad_biases = []
        for i in range(len(self.weights) - 1, -1, -1):
            grad_weights.append(np.dot(errors, activations.T))
            grad_biases.append(errors)
            errors = np.dot(self.weights[i].T, errors)
            
        # Reverse the gradients so they can be applied in the correct order
        grad_weights = grad_weights[::-1]
        grad_biases = grad_biases[::-1]
        
        # Update the weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * grad_weights[i]
            self.biases[i] += learning_rate * grad_biases[i]
