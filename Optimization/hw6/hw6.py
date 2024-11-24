import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Dimensions and parameters
n = 10
alpha = 1.0  # Ensures positive definiteness
T = 1000     # Number of iterations

# Generate symmetric positive definite matrix Q
A = np.random.randn(n, n)
Q = A.T @ A + alpha * np.eye(n)

# Generate random vector q
q = np.random.randn(n)

# Compute the optimal solution x_star
x_star = -np.linalg.solve(Q, q)

# Objective function and its gradient
def f(x):
    return 0.5 * x.T @ Q @ x + q.T @ x

def grad_f(x):
    return Q @ x + q

# Initial point x0
x0 = np.random.randn(n)

# Gradient descent with eta_t = 1 / t
x = x0.copy()
f_vals_1 = []
for t in range(1, T + 1):
    eta_t = 1.0 / t
    x -= eta_t * grad_f(x)
    f_vals_1.append(f(x) - f(x_star))

# Gradient descent with eta_t = 1 / sqrt(t)
x = x0.copy()
f_vals_2 = []
for t in range(1, T + 1):
    eta_t = 1.0 / np.sqrt(t)
    x -= eta_t * grad_f(x)
    f_vals_2.append(f(x) - f(x_star))

# Plotting the results for diminishing step sizes
plt.figure(figsize=(10, 6))
plt.plot(f_vals_1, label='ηₜ = 1 / t')
plt.plot(f_vals_2, label='ηₜ = 1 / √t')
plt.xlabel('Iteration')
plt.ylabel('f(xₜ) - f(x*)')
plt.title('Gradient Descent with Diminishing Step Sizes')
plt.legend()
plt.show()

# Gradient descent with fixed step sizes
etas = [0.1, 0.5, 1.0, 1.5, 2.0]
plt.figure(figsize=(10, 6))

for eta in etas:
    x = x0.copy()
    f_vals_fixed = []
    diverged = False
    for t in range(1, T + 1):
        x -= eta * grad_f(x)
        f_diff = f(x) - f(x_star)
        f_vals_fixed.append(f_diff)
        if f_diff > 1e5:  # Divergence threshold
            print(f'Divergence detected for η = {eta} at iteration {t}')
            diverged = True
            break
    plt.plot(f_vals_fixed, label=f'η = {eta}')
    if diverged:
        plt.axvline(x=t, color='k', linestyle='--')

plt.xlabel('Iteration')
plt.ylabel('f(xₜ) - f(x*)')
plt.title('Gradient Descent with Fixed Step Sizes')
plt.legend()
plt.show()
