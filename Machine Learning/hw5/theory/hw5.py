import numpy as np

# Create the covariance matrix Σ
Sigma = np.array([
    [0.71, -0.43, 0.43, 0],
    [-0.43, 0.46, -0.26, 0],
    [0.43, -0.26, 0.46, 0],
    [0, 0, 0, 0.2]
])

# Create the inverse covariance matrix Q
Q = np.array([
    [5, 3, -3, 0],
    [3, 5, 0, 0],
    [-3, 0, 5, 0],
    [0, 0, 0, 5]
])

# Verify that Q is indeed the inverse of Sigma
print("Verification that Q is the inverse of Sigma:")
print(np.allclose(np.dot(Sigma, Q), np.eye(4)))  # Should return True if Q is indeed the inverse

# Check correlation between X3 and X4
correlation_X3_X4 = Sigma[2,3]  # Remember Python uses 0-based indexing
print(f"Correlation between X3 and X4: {correlation_X3_X4}")

# The answer is: No, X3 and X4 are not correlated because their covariance is 0 