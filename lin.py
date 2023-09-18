import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Initialize weights and bias
theta0 = np.random.randn()
theta1 = np.random.randn()

# Define learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
for iteration in range(num_iterations):
    # Calculate predictions
    y_pred = theta0 + theta1 * X
    
    # Calculate gradients
    gradient0 = -np.mean(y - y_pred)
    gradient1 = -np.mean((y - y_pred) * X)
    
    # Update weights and bias
    theta0 -= learning_rate * gradient0
    theta1 -= learning_rate * gradient1

# Print the final values of theta0 and theta1
print("Theta0:", theta0)
print("Theta1:", theta1)

# Plot the data and the linear regression line
plt.scatter(X, y, label='Data')
plt.plot(X, theta0 + theta1 * X, 'r-', label='Linear Regression')
plt.xlabel('inputs ')
plt.ylabel('outputs')
plt.legend()
plt.show()
