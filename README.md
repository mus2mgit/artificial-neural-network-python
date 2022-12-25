# ANN

A Python class for creating and training artificial neural networks (ANNs).

## **Installation**

To use the ANN class, you will need to have NumPy installed. You can install NumPy using pip:

```
pip install numpy
```

## **Usage**

To use the ANN class, you will need to import it and create an instance of the class. The constructor takes a list of integers representing the number of neurons in each layer of the ANN, starting with the input layer and ending with the output layer. For example, to create an ANN with 2 input neurons, 3 hidden neurons, and 1 output neuron, you would call **`ANN([2, 3, 1])`**.

```
import numpy as np
from ANN import ANN

ann = ANN([2, 3, 1])
```

The ANN class has three methods:

### **`sigmoid(x)`**

The **`sigmoid`** method implements the sigmoid function, which is used as the activation function for the neurons in the ANN. It takes a single argument, **`x`**, and returns the value of the sigmoid function applied to **`x`**.

### **`forward_pass(inputs)`**

The **`forward_pass`** method performs a forward pass through the ANN, taking a single argument, **`inputs`**, which should be a NumPy array with the same number of elements as the number of input neurons in the ANN. It returns a NumPy array with the same number of elements as the number of output neurons in the ANN, representing the outputs of the ANN for the given inputs.

### **`train(inputs, targets, learning_rate)`**

The **`train`** method trains the ANN on a single training example, updating the weights and biases of the ANN based on the error between the predicted outputs and the desired targets. It takes three arguments:

- **`inputs`**: a NumPy array with the same number of elements as the number of input neurons in the ANN.
- **`targets`**: a NumPy array with the same number of elements as the number of output neurons in the ANN.
- **`learning_rate`**: a float representing the learning rate for the ANN.

## **Example**

Here is an example of how to use the ANN class to train an ANN to perform binary classification on a simple dataset:

```
import numpy as np
from ANN import ANN

# Create the ANN
ann = ANN([2, 3, 1])

# Define the training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the ANN
for i in range(10000):
    ann.train(inputs[i % 4], targets[i % 4], 0.1)

# Test the ANN
print(ann.forward_pass(np.array([0, 0]))) # should output [0]
print(ann.forward_pass(np.array([0, 1]))) # should output [1]
print(ann.forward_pass(np.array([1, 0])))
print(ann.forward_pass(np.array([1, 0]))) # should output [1]
print(ann.forward_pass(np.array([1, 1]))) # should output [0]
```

This example creates an ANN with 2 input neurons and 1 output neuron, and trains it on a simple XOR dataset using the train method. It then uses the forward_pass method to test the ANN on the same input data and prints the outputs.

## **Notes**

- The ANN class is designed to work with binary classification tasks. If you want to use it for a different type of task, you may need to modify the **`sigmoid`** function and/or the way that the error is calculated and propagated through the network.
- The ANN class uses stochastic gradient descent to update the weights and biases of the network. This means that the weights and biases are updated based on the error for a single training example at a time, rather than the error for the entire dataset. This can make the training process faster, but it also means that the ANN may not converge to a global minimum for the error function. You may need to experiment with different learning rates and/or training algorithms to achieve good results.
