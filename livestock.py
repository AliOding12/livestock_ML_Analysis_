import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path_livestock = 'F:/paid/LiveStockData.csv'
livestock_data = pd.read_excel(file_path_livestock)

# Separate features (X) and target (y)
X_multivar = livestock_data[['Cattle (thousands)', 'Calves (thousands)', 'Pigs (thousands)', 'Lambs (thousands)']]
y_multivar = livestock_data['Expenses (1000*dollars)']

# Add a column of ones to X to account for the intercept term (theta_0)
X_b_multivar = np.c_[np.ones(X_multivar.shape[0]), X_multivar]

# Initialize theta parameters for multiple features (number of features + 1 for intercept)
theta_multivar = np.zeros(X_b_multivar.shape[1])

# Define the cost function for multivariate linear regression
def compute_cost_multivariate(X, y, theta):
    m = len(y)  # Number of training examples
    predictions = X.dot(theta)  # Predictions for all examples
    errors = predictions - y  # Difference between predictions and actual values
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)  # Mean squared error
    return cost

# Set hyperparameters for gradient descent
alpha = 0.01
num_iters = 1500

# Implement gradient descent for multivariate linear regression
def gradient_descent_multivariate(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []  # To record the cost in each iteration
    theta = theta.copy()  # Avoid modifying the original theta array
    
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)  # Update theta
        cost_history.append(compute_cost_multivariate(X, y, theta))  # Save the cost for each iteration
    
    return theta, cost_history

# Run gradient descent
theta_optimal, cost_history_multivar = gradient_descent_multivariate(X_b_multivar, y_multivar, theta_multivar, alpha, num_iters)

# Output the optimized theta parameters
print("Optimized Theta:", theta_optimal)

# Plotting the cost function over iterations to visualize convergence
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), cost_history_multivar, color='blue')
plt.title("Cost Function J(θ) Over Iterations for Multivariate Linear Regression")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost J(θ)")
plt.grid(True)
plt.show()

# Prediction function using the optimized theta
def predict_expenses(features, theta):
   
    # Add intercept term to the feature array
    features_with_intercept = np.insert(features, 0, 1)
    # Calculate prediction
    prediction = np.dot(features_with_intercept, theta)
    return prediction

# Example prediction (replace values with desired features to predict)
example_features = [10, 5, 2, 15]  # Example: [Cattle, Calves, Pigs, Lambs]
predicted_expenses = predict_expenses(example_features, theta_optimal)
print("Predicted Expenses:", predicted_expenses)
