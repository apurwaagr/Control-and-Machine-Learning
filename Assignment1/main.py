
import numpy as np


max_iteration = 50000

def objective(x):
    term1 = 100 * (x[1] - x[0]**2)**2
    term2 = (1 - x[0])**2
    return term1 + term2

def gradient(x):

    term1 = np.clip(-400 * x[0] * (x[1] - x[0]**2), -1e6, 1e6)
    term2 = np.clip(2 * (x[0] - 1), -1e6, 1e6)
    term3 = np.clip(200 * (x[1] - x[0]**2), -1e6, 1e6)
    return np.array([term1 + term2, term3])

def hessian(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

# Gradient descent method with constant step sizes: 0.1, 0.01, 0.001, 0.0001.
def gradient_descent_constant_step(step_size):
    x = np.array([0, 0])
    epsilon = 1e-8

    for i in range(max_iteration):
        grad = gradient(x)
        x = x - step_size * grad

        if np.linalg.norm(grad) < epsilon:
            break

    return x

step_sizes = [0.1, 0.01, 0.001, 0.0001]

for step_size in step_sizes:
    result = gradient_descent_constant_step(step_size)
    if step_size == 0.1:
        print("Gradient Descent Method ")
        print("----------------------- ")
        print("-> Method starts with fixed step size and updates the current point in the direction of negative gradient")
    print(f"Constant Step Size: {step_size})= {result}")
    if step_size == 0.0001:
        print("-> Larger the step size algorithm converges faster but overshoot the optimal solution")
        print("-> Smaller the step size, algorithm converges slower but provide more accurate solution")

# Gradient descent method with backtracking line search.
def gradient_descent_backtracking_line_search():
    x = np.array([0, 0])
    epsilon = 1e-8
    step_size = 1.0
    shrinkage_factor = 0.5
    sufficient_decrease = 0.5

    for i in range(max_iteration):
        grad = gradient(x)
        while objective(x - step_size * grad) > objective(x) - step_size * sufficient_decrease * np.dot(grad, grad):
            step_size *= shrinkage_factor

        x = x - step_size * grad

        if np.linalg.norm(grad) < epsilon:
            break

    return x

result = gradient_descent_backtracking_line_search()
print("--------------------------------------------")
print(f"Gradient Descent for Backtracking Line Search)= {result}")
print("-> This method dynamically adjusts the step size and ensure convergence and improve efficiency")
print("-> Algorithm starts with larger step size and gradually reduces until the Armijo condition is satisfied")
print("--------------------------------------------")

# Classic Newton method (i.e. step size is 1).
def newton_method():
    x = np.array([0, 0])
    epsilon = 1e-8

    for i in range(max_iteration):
        grad = gradient(x)
        hess = hessian(x)
        direction = -np.linalg.inv(hess).dot(grad)
        x = x + direction

        if np.linalg.norm(grad) < epsilon:
            break

    return x

result = newton_method()
print("---------------------- ")
print(f"Classic Newton Method= {result}")
print("-> This method utilises both the gradient and Hessian matrix to determine the updated direction")
print("-> Optimal solution is reached in fewer iterations")
print("-> This problem is quadratic hence exact solution (1,1)T is reached in just few iterations")
print("---------------------- ")

# Newton method with backtracking line search.
def newton_method_backtracking_line_search():
    x = np.array([0, 0])
    epsilon = 1e-8
    step_size = 1.0
    shrinkage_factor = 0.5
    sufficient_decrease = 0.5

    for i in range(max_iteration):
        grad = gradient(x)
        hess = hessian(x)

        while objective(x - step_size * np.linalg.inv(hess).dot(grad)) > objective(x) - step_size * sufficient_decrease * np.dot(grad, np.linalg.inv(hess)).dot(grad):
            step_size *= shrinkage_factor

        direction = -np.linalg.inv(hess).dot(grad)
        x = x + step_size * direction

        if np.linalg.norm(grad) < epsilon:
            break

    return x

result = newton_method_backtracking_line_search()
print("------------------------------------------ ")
print(f"Newton Method for Backtracking Line Search)= {result}")
print("-> This method combines the advantages of Newton Method and backtracking line search")
print("-> It uses the Newton update equation for determining the update direction and adjusts the step size through line search")
print("------------------------------------------ ")
