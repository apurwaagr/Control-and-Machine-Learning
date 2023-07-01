import numpy as np
from scipy.optimize import minimize

def objective(x):
    x1, x2 = x
    return 100*(x2 - x1**2)**2 + (x1 - 1)**2

def gradient(x):
    x1, x2 = x
    grad_x1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
    grad_x2 = 200*(x2 - x1**2)
    return np.array([grad_x1, grad_x2])

def dfp_update(Hk, g, s, y, *args):
    rho = 1 / np.dot(y, s)
    A = np.eye(len(g)) - rho * np.outer(s, y)
    B = np.eye(len(g)) - rho * np.outer(y, s)
    return np.dot(np.dot(A, Hk), B) + rho * np.outer(s, s)

def bfgs_update(Hk, g, s, y, *args):
    rho = 1 / np.dot(y, s)
    A = np.eye(len(g)) - rho * np.outer(s, y)
    B = np.eye(len(g)) - rho * np.outer(y, s)
    return np.dot(np.dot(A, Hk), B) + rho * np.outer(g, g)

def quasi_newton_optimization(update_formula, x0, *args):
    max_iter = 50000
    stopping_criterion = 1e-8

    def callback(xk):
        nonlocal iterations
        iterations.append(xk)

    iterations = [x0]
    result = minimize(objective, x0, method='BFGS', jac=gradient, hessp=lambda x, p: update_formula((Hk, gradient(x), s, y)), callback=callback,
                      options={'gtol': stopping_criterion, 'maxiter': max_iter})

    return result.x, iterations

# Test the different quasi-Newton methods
x0 = np.array([0, 0])

# DFP formula for Hk
result_dfp_hk, iterations_dfp_hk = quasi_newton_optimization(dfp_update, x0)

# BFGS formula for Hk
result_bfgs_hk, iterations_bfgs_hk = quasi_newton_optimization(bfgs_update, x0)

# DFP formula for Bk
def dfp_hessp(x, p):
    g = gradient(x)
    s = iterations_dfp_bk[-2] - iterations_dfp_bk[-3]
    y = g - gradient(iterations_dfp_bk[-2])
    return dfp_update(p, g, s, y, x, p)

result_dfp_bk, iterations_dfp_bk = quasi_newton_optimization(dfp_hessp, x0)

# BFGS formula for Bk
def bfgs_hessp(x, p):
    g = gradient(x)
    s = iterations_bfgs_bk[-2] - iterations_bfgs_bk[-3]
    y = g - gradient(iterations_bfgs_bk[-2])
    return bfgs_update(p, g, s, y, x, p)

result_bfgs_bk, iterations_bfgs_bk = quasi_newton_optimization(bfgs_hessp, x0)

# Output results
print("DFP formula for Hk:")
print("Optimization result:", result_dfp_hk)
print("Objective value at the result:", objective(result_dfp_hk))
print("Number of iterations:", len(iterations_dfp_hk)-1)
print("Iterations:")
for i, iteration in enumerate(iterations_dfp_hk):
    print(f"Iteration {i+1}: {iteration}")
print()

print("BFGS formula for Hk:")
print("Optimization result:", result_bfgs_hk)
print("Objective value at the result:", objective(result_bfgs_hk))
print("Number of iterations:", len(iterations_bfgs_hk)-1)
print("Iterations:")
for i, iteration in enumerate(iterations_bfgs_hk):
    print(f"Iteration {i+1}: {iteration}")
print()

print("DFP formula for Bk:")
print("Optimization result:", result_dfp_bk)
print("Objective value at the result:", objective(result_dfp_bk))
print("Number of iterations:", len(iterations_dfp_bk)-1)
print("Iterations:")
for i, iteration in enumerate(iterations_dfp_bk):
    print(f"Iteration {i+1}: {iteration}")
print()

print("BFGS formula for Bk:")
print("Optimization result:", result_bfgs_bk)
print("Objective value at the result:", objective(result_bfgs_bk))
print("Number of iterations:", len(iterations_bfgs_bk)-1)
print("Iterations:")
for i, iteration in enumerate(iterations_bfgs_bk):
    print(f"Iteration {i+1}: {iteration}")
