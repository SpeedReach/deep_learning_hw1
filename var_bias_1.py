import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy.optimize import curve_fit

mat_data = scipy.io.loadmat('data.mat')
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()

# Number of random samples
num_samples = 30
num_iterations = 200

# Initialize an array to store the fitted theta_0 and theta_1 values
theta_0_values = []
theta_1_values = []

def parabola(v, theta_0, theta_1):
    return theta_0 + theta_1 * v

for _ in range(num_iterations):
    # Randomly select 30 data samples
    random_indices = np.random.choice(len(x), size=num_samples, replace=False)
    x_sample = x[random_indices]
    y_sample = y[random_indices]
    
    # Perform linear regression (first-order polynomial)
    params, _ = curve_fit(parabola, x_sample, y_sample)
    
    # Extract the parameters
    theta_0, theta_1 = params
    
    # Store the parameters
    theta_0_values.append(theta_0)
    theta_1_values.append(theta_1)

# Plot the 200 lines
for i in range(num_iterations):
    plt.plot(x, theta_0_values[i] + theta_1_values[i]*x, color='grey', alpha=0.1)

# Plot the original data
plt.scatter(x, y, color='red', label='Original Data')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines')
plt.legend()
plt.show()

