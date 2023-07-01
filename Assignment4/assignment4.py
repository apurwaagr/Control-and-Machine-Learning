import numpy as np
import matplotlib.pyplot as plt

# Training data
x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Neural network details
input_layer_size = 2
hidden_layer_size = 100
output_layer_size = 2
num_iterations = 1000


# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


# Forward propagation
def forward_propagation(x1, x2, theta1, theta2, activation):
    m = x1.shape[0]  # Number of training examples
    a1 = np.vstack((x1, x2))  # Input features
    a1 = np.vstack((np.ones(m), a1))  # Add bias term to input layer
    z2 = np.dot(theta1, a1)
    a2 = activation(z2)
    a2 = np.vstack((np.ones(m), a2))  # Add bias term to hidden layer
    z3 = np.dot(theta2, a2)
    a3 = sigmoid(z3)
    return a1, a2, a3


# Backpropagation
def back_propagation(x1, x2, y, a1, a2, a3, theta2, activation):
    m = x1.shape[0]  # Number of training examples
    delta3 = a3 - y
    delta2 = np.dot(theta2.T, delta3) * (a2 * (1 - a2))
    delta2 = delta2[1:, :]  # Remove bias term
    grad1 = np.dot(delta2, a1.T) / m
    grad2 = np.dot(delta3, a2.T) / m
    return grad1, grad2


# Loss function
def loss_function(a3, y):
    m = y.shape[1]  # Number of training examples
    loss = (-1 / m) * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    return loss

# Training function
def train_neural_network(x1, x2, y, activation, learning_rate, optimizer=None, momentum=0.9, nesterov=False):
    # Initialize neural network parameters
    theta1 = np.random.randn(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.randn(output_layer_size, hidden_layer_size + 1)

    # Initialize velocity for momentum or Nesterov
    v1 = np.zeros_like(theta1)
    v2 = np.zeros_like(theta2)

    # Training loop
    losses = np.zeros(num_iterations)
    for i in range(num_iterations):
        # Forward propagation
        a1, a2, a3 = forward_propagation(x1, x2, theta1, theta2, activation)

        # Backpropagation
        grad1, grad2 = back_propagation(x1, x2, y, a1, a2, a3, theta2, activation)

        # Update parameters using specified optimizer
        if optimizer == "momentum":
            v1 = momentum * v1 - learning_rate * grad1
            v2 = momentum * v2 - learning_rate * grad2
            theta1 += v1
            theta2 += v2
        elif optimizer == "nesterov":
            v1_prev = v1
            v2_prev = v2
            v1 = momentum * v1 - learning_rate * grad1
            v2 = momentum * v2 - learning_rate * grad2
            theta1 += -momentum * v1_prev + (1 + momentum) * v1
            theta2 += -momentum * v2_prev + (1 + momentum) * v2
        else:
            theta1 -= learning_rate * grad1
            theta2 -= learning_rate * grad2

        # Calculate loss function
        losses[i] = loss_function(a3, y)

    return losses


# Training the neural networks with different optimizers
learning_rate = 0.05

# Standard SGD
losses_sgd = train_neural_network(x1, x2, y, sigmoid, learning_rate, optimizer=None)

# SGD with momentum
losses_momentum = train_neural_network(x1, x2, y, sigmoid, learning_rate, optimizer="momentum")

# Nesterov accelerated SGD
losses_nesterov = train_neural_network(x1, x2, y, sigmoid, learning_rate, optimizer="nesterov")

# Plot loss function versus iteration number
plt.figure()
plt.plot(range(1, num_iterations + 1), losses_sgd, label="SGD")
plt.plot(range(1, num_iterations + 1), losses_momentum, label="Momentum")
plt.plot(range(1, num_iterations + 1), losses_nesterov, label="Nesterov")
plt.title(f'{sigmoid.__name__} Activation Function (Learning Rate: {learning_rate})')
plt.xlabel('Iteration Number')
plt.ylabel('Value of Cost Function')
plt.legend()
plt.show()
