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
def parabola(v, theta_0, theta_1, theta_2):
    return theta_0 + theta_1 * v + theta_2 * v**2

# Perform the curve fitting using least squares
params, covariance = curve_fit(parabola, x, y)

# Extract the parameters
theta_0, theta_1, theta_2 = params

# Print the parameters
print(f'Theta_0: {theta_0}')
print(f'Theta_1: {theta_1}')
print(f'Theta_2: {theta_2}')

# Calculate the predicted y values using the fitted parameters
y_pred = parabola(x, *params)

# Calculate the mean squared error
mse = np.mean((y - y_pred)**2)

# Print the mean squared error
print(f'Mean Squared Error: {mse}') #0.01574491993120757

# Generate the least squares parabola
y_ls = theta_0 + x*theta_1 + x**2*theta_2
plt.plot(x, y_ls, label=f'least Squares Parabola: y = {theta_0:.2f} + {theta_1:.2f}x + {theta_2:.2f}x^2', color='red')

plt.legend()
plt.show()

