import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data (assuming a dataset with multiple features in 'data_multivariate.csv.xlsx')
file_path = 'F:/paid/data01.csv'  # Replace with actual file path
data = pd.read_excel(file_path)

# Extract features (all columns except the last one) and target (last column)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Add a column of ones to X for the intercept term
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# Initialize parameters
theta = np.zeros(X.shape[1])
m = len(y)

# Define the cost function
def compute_cost_multi(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Define the gradient descent function for multivariate regression
def gradient_descent_multi(X, y, theta, alpha, num_iters=1000, tolerance=1e-6):
    J_history = []
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        
        # Compute and store the cost
        cost = compute_cost_multi(X, y, theta)
        J_history.append(cost)
        
        # Check for convergence
        if i > 0 and abs(J_history[-2] - J_history[-1]) < tolerance:
            break
    
    return theta, J_history

# Define a range of learning rates for analysis
learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003]
iterations_needed = []

# Plot the cost vs. iteration for each learning rate
plt.figure(figsize=(10, 6))
for alpha in learning_rates:
    theta = np.zeros(X.shape[1])  # Reinitialize theta for each alpha
    _, J_history = gradient_descent_multi(X, y, theta, alpha)
    iterations_needed.append(len(J_history))
    plt.plot(range(len(J_history)), J_history, label=f'α = {alpha}')

plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Effect of Learning Rate on Convergence (Multivariate)')
plt.legend()
plt.show()

# Display the number of iterations needed for each learning rate
for alpha, iters in zip(learning_rates, iterations_needed):
    print(f"Learning rate α = {alpha}: Converged in {iters} iterations")

# 1. Predicted vs. Actual Plot
# Calculate predictions using the learned theta values from gradient descent
theta, _ = gradient_descent_multi(X, y, np.zeros(X.shape[1]), alpha=0.01)  # Adjust alpha as needed
predictions = X.dot(theta)

plt.figure(figsize=(8, 6))
plt.scatter(y, predictions, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r', linewidth=2)  # 45-degree line for reference
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Predicted vs. Actual Target Values')
plt.show()

# 2. Individual Feature Fit Plots
fig, axes = plt.subplots(1, X.shape[1] - 1, figsize=(15, 5))
fig.suptitle('Individual Feature Fit with Multivariate Regression Predictions')

# Loop through each feature (excluding intercept)
for i in range(1, X.shape[1]):
    axes[i - 1].scatter(data.iloc[:, i - 1], y, color='blue', label='Actual')
    axes[i - 1].scatter(data.iloc[:, i - 1], X[:, i] * theta[i] + theta[0], color='red', label='Predicted', alpha=0.5)
    axes[i - 1].set_xlabel(f'Feature {i}')
    axes[i - 1].set_ylabel('Target')
    axes[i - 1].legend()

plt.tight_layout()
plt.show()

# 3. Residual Plot
residuals = y - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
