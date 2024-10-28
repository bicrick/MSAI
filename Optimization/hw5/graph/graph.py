import numpy as np
import matplotlib.pyplot as plt

def plot_feasible_region():
    # Create a grid of points
    x1 = np.linspace(-2, 5, 300)
    x2 = np.linspace(-2, 5, 300)
    X1, X2 = np.meshgrid(x1, x2)

    # Define the constraints
    constraint1 = X1**2 - 2*X1 + X2**2 <= 0
    constraint2 = 0.5*X1**2 - 2*X1 - X2 + 2 <= 0
    constraint3 = X1 - X2 <= 0

    # Combine all constraints
    feasible_region = constraint1 & constraint2 & constraint3

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(feasible_region.astype(int), 
               extent=[x1.min(), x1.max(), x2.min(), x2.max()], 
               origin='lower', cmap='Blues', alpha=0.3)

    # Plot the constraint boundaries
    plt.contour(X1, X2, X1**2 - 2*X1 + X2**2, [0], colors='r', linestyles='dashed')
    plt.contour(X1, X2, 0.5*X1**2 - 2*X1 - X2 + 2, [0], colors='g', linestyles='dashed')
    plt.contour(X1, X2, X1 - X2, [0], colors='b', linestyles='dashed')

    # Set labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Feasible Region')

    # Add a legend
    plt.plot([], [], 'r--', label='x1^2 - 2x1 + x2^2 ≤ 0')
    plt.plot([], [], 'g--', label='0.5x1^2 - 2x1 - x2 + 2 ≤ 0')
    plt.plot([], [], 'b--', label='x1 - x2 ≤ 0')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Call the function to create the plot
plot_feasible_region()
