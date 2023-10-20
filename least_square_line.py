import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# load data
mat_data = scipy.io.loadmat('data.mat')
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()

# plot data
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')


n = len(x)
def line(v, theta_0, theta_1):
    return theta_0 + theta_1 * v

# Perform the curve fitting using least squares
params, covariance = curve_fit(line, x, y)

# Extract the parameters
theta_0, theta_1 = params

# Print the parameters
print(f'Theta_0: {theta_0}')
print(f'Theta_1: {theta_1}')

# Calculate the predicted y values using the fitted parameters
y_pred = line(x, *params)

# Calculate the mean squared error
mse = np.mean((y - y_pred)**2)

# Print the mean squared error
print(f'Mean Squared Error (MSE): {mse}') #0.20580596682517396

# Generate the least squares line
y_ls = theta_0 + x*theta_1
plt.plot(x, y_ls, label=f'least square line: y = {theta_0:.2f} + {theta_1:.2f}x', color='red')

plt.legend()
plt.show()

