import numpy as np
import matplotlib.pyplot as plt

def plot_all_ellipses():
    # Create a circle of points
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Define matrices
    Q1 = np.array([[1/9, 0],
                   [0, 1]])
    A = np.array([[3, 0],
                  [0, 1]])
    Q2 = np.array([[1, 0.9],
                   [0.9, 1]])
    
    # Get the square root of Q inverse for each case
    Q1_sqrt_inv = np.linalg.inv(np.sqrt(Q1))
    Q2_sqrt_inv = np.linalg.inv(np.sqrt(Q2))
    
    # Transform circles into ellipses
    ellipse_a = Q1_sqrt_inv @ circle
    ellipse_b = np.linalg.inv(A) @ (Q1_sqrt_inv @ circle)
    ellipse_c = Q2_sqrt_inv @ circle
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot (a) - original ellipse with Q = [[1/9, 0], [0, 1]]
    ax1.plot(ellipse_a[0, :], ellipse_a[1, :], 'b-', label='Ellipse')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('(a) Original Ellipse\nx^T Q x ≤ 1, Q = diag(1/9, 1)')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.legend()
    
    # Plot (b) - transformed ellipse in y-coordinates
    ax2.plot(ellipse_b[0, :], ellipse_b[1, :], 'r-', label='Transformed Ellipse')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.set_xlabel('y₁')
    ax2.set_ylabel('y₂')
    ax2.set_title('(b) Transformed Ellipse\nin y-coordinates')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.legend()
    
    # Plot (c) - new ellipse with Q = [[1, 0.9], [0.9, 1]]
    ax3.plot(ellipse_c[0, :], ellipse_c[1, :], 'g-', label='Ellipse')
    ax3.grid(True)
    ax3.axis('equal')
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_title('(c) New Ellipse\nx^T Q x ≤ 1, Q = [[1, 0.9], [0.9, 1]]')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def plot_spherical_transformation():
    # Create a circle of points (unit circle)
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Define Q matrix
    Q = np.array([[1, 0.9],
                  [0.9, 1]])
    
    # Compute the eigenvalues and eigenvectors of Q
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    
    # Compute the square root of Q (matrix square root)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    Q_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    # Compute the inverse square root of Q to generate points on the ellipse
    Q_inv_sqrt = eigenvectors @ np.diag(1/np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Transform circle into ellipse using Q_inv_sqrt
    ellipse = Q_inv_sqrt @ circle
    
    # Transform ellipse into sphere using Q_sqrt (A matrix)
    sphere = Q_sqrt @ ellipse
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original ellipse
    ax1.plot(ellipse[0, :], ellipse[1, :], 'g-', label='Original Ellipse')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('(d) Original Ellipse\n$x^T Q x \\leq 1$')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.legend()
    
    # Plot transformed sphere
    ax2.plot(sphere[0, :], sphere[1, :], 'r-', label='Transformed to Sphere')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.set_xlabel('$y_1$')
    ax2.set_ylabel('$y_2$')
    ax2.set_title('(d) Transformed to Sphere\n$y^T y \\leq 1$')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print the A matrix
    print("\nThe transformation matrix A that makes the ellipse spherical is:")
    print(Q_sqrt)

if __name__ == "__main__":
    plot_all_ellipses()  # This will show parts a, b, c
    plot_spherical_transformation()  # This will show part d
