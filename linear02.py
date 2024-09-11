import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting toolkit

# Load and rename columns for clarity
file_path = 'F:/paid/data02.csv'
data = pd.read_csv(file_path)
data.columns = ['Size', 'Rooms', 'Price']

# Step 1: Scatter Plot of Each Feature vs Target
sns.pairplot(data, y_vars="Price", x_vars=["Size", "Rooms"], height=3, aspect=1.2)
plt.suptitle('Scatter Plot of Each Feature vs Target', y=1.02)
plt.show()

# Step 2: Feature Normalization
X = data[['Size', 'Rooms']].copy()
X = (X - X.mean()) / X.std()
y = data['Price']
y = (y - y.mean()) / y.std()

# Add intercept term to normalized features
X = np.concatenate([np.ones((X.shape[0], 1)), X.values], axis=1)

# Initialize parameters
theta = np.zeros(X.shape[1])

# Define cost function for multivariate linear regression
def compute_cost_multi(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Define gradient descent for multivariate regression
def gradient_descent_multi(X, y, theta, alpha, num_iters=1000, tolerance=1e-6):
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        cost = compute_cost_multi(X, y, theta)
        J_history.append(cost)
        
        # Convergence check
        if i > 0 and abs(J_history[-2] - J_history[-1]) < tolerance:
            break
    
    return theta, J_history

# Step 3: Run Gradient Descent with Different Learning Rates and Plot Convergence
learning_rates = [0.01, 0.003, 0.001, 0.0003, 0.0001]
iterations_needed = []

plt.figure(figsize=(10, 6))
for alpha in learning_rates:
    theta_initial = np.zeros(X.shape[1])
    _, J_history = gradient_descent_multi(X, y, theta_initial, alpha)
    iterations_needed.append(len(J_history))
    plt.plot(range(len(J_history)), J_history, label=f'α = {alpha}')

plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Convergence of Cost Function for Normalized Features')
plt.legend()
plt.show()

# Display the number of iterations needed for each learning rate with normalized data
for alpha, iters in zip(learning_rates, iterations_needed):
    print(f"Learning rate α = {alpha}: Converged in {iters} iterations")

# Step 4: Visualize the Cost Function for θ0 and θ1 (Intercept and First Feature)
theta_optimal, _ = gradient_descent_multi(X, y, np.zeros(X.shape[1]), alpha=0.01)  # Using optimal α

# Set up grid for θ0 and θ1
theta0_vals = np.linspace(-1, 1, 100)
theta1_vals = np.linspace(-1, 1, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Calculate cost function for each pair of θ0 and θ1
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        theta_temp = np.array([theta0, theta1, theta_optimal[2]])  # Fix θ2 for simplicity
        J_vals[i, j] = compute_cost_multi(X, y, theta_temp)

# Surface and Contour Plots
fig = plt.figure(figsize=(14, 6))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')  # Set up 3D projection
theta0_vals_mesh, theta1_vals_mesh = np.meshgrid(theta0_vals, theta1_vals)
ax1.plot_surface(theta0_vals_mesh, theta1_vals_mesh, J_vals.T, cmap='viridis', edgecolor='none')
ax1.set_xlabel('θ0')
ax1.set_ylabel('θ1 (First Feature)')
ax1.set_zlabel('Cost J(θ)')
ax1.set_title('Surface Plot of Cost Function')

# Contour plot
ax2 = fig.add_subplot(122)
ax2.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 20), cmap='viridis')
ax2.scatter(theta_optimal[0], theta_optimal[1], color='red', marker='x', s=50, label='Optimal θ')
ax2.set_xlabel('θ0')
ax2.set_ylabel('θ1 (First Feature)')
ax2.set_title('Contour Plot of Cost Function')
ax2.legend()
plt.show()
# Add linear regression script for data02
# Optimize regression scripts for performance
