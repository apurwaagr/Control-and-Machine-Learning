import numpy as np
import matplotlib.pyplot as plt

# Training data
x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.5, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.25, 0.3, 0.6, 0.2, 0.4, 0.6])
y = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Neural network settings
input_layer_size = 2
hidden_layer_size = 4
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
def train_neural_network(x1, x2, y, activation, learning_rate):
    # Initialize neural network parameters
    theta1 = np.random.randn(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.randn(output_layer_size, hidden_layer_size + 1)

    # Training loop
    losses = np.zeros(num_iterations)
    for i in range(num_iterations):
                # Forward propagation
        a1, a2, a3 = forward_propagation(x1, x2, theta1, theta2, activation)

        # Backpropagation
        grad1, grad2 = back_propagation(x1, x2, y, a1, a2, a3, theta2, activation)

        # Update parameters
        theta1 -= learning_rate * grad1
        theta2 -= learning_rate * grad2

        # Calculate loss function
        losses[i] = loss_function(a3, y)

    # Plot loss function versus iteration number
    plt.figure()
    plt.plot(range(1, num_iterations + 1), losses)
    plt.title(f'{activation.__name__} Activation Function (Learning Rate: {learning_rate})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Visualization of output
    visualize_output(theta1, theta2, activation)

# Visualization of output
def visualize_output(theta1, theta2, activation):
    # Generate grid points for visualization
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    xx_flatten = xx.ravel()
    yy_flatten = yy.ravel()

    # Forward propagation for grid points
    a1, a2, a3 = forward_propagation(xx_flatten, yy_flatten, theta1, theta2, activation)

    # Reshape output to match grid dimensions
    a3 = a3[0, :].reshape(xx.shape)

    # Plot decision boundaries
    plt.figure()
    plt.contourf(xx, yy, a3, levels=[-1, 0, 1], colors=['b', 'r'], alpha=0.2)
    plt.scatter(x1[:6], x2[:6], c='b', marker='o', label='Category A')
    plt.scatter(x1[6:], x2[6:], c='r', marker='x', label='Category B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'{activation.__name__} Activation Function Decision Boundaries')
    plt.legend()
    plt.show()

# Training the neural networks
learning_rates = [0.1, 0.05, 0.01]

# Sigmoid Activation Function
for lr in learning_rates:
    train_neural_network(x1, x2, y, sigmoid, lr)

# Hyperbolic Tangent Activation Function
for lr in learning_rates:
    train_neural_network(x1, x2, y, np.tanh, lr)

