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
def line(v, theta_0, theta_1, theta_2, theta_3, theta_4):
    return theta_0 + theta_1 * v + theta_2 * v**2 + theta_3 * v**3 + theta_4 * v**4

# Perform the curve fitting using least squares
params, covariance = curve_fit(line, x, y)

# Extract the parameters
theta_0, theta_1, theta_2, theta_3, theta_4 = params

# Print the parameters
print(f'Theta_0: {theta_0}')
print(f'Theta_1: {theta_1}')
print(f'Theta_2: {theta_2}')
print(f'Theta_3: {theta_3}')
print(f'Theta_4: {theta_4}')

# Calculate the predicted y values using the fitted parameters
y_pred = line(x, *params)

# Calculate the mean squared error
mse = np.mean((y - y_pred)**2)

# Print the mean squared error
print(f'Mean Squared Error: {mse}') # 0.010413051044174369

# Generate the least square quartic curve
y_ls = theta_0 + x*theta_1 + x**2*theta_2 + x**3*theta_3 + x**4*theta_4
plt.plot(x, y_ls, label=f'least square quartic curve: y = {theta_0:.2f} + {theta_1:.2f}x + {theta_2:.2f}x^2 + {theta_3:.2f}x^3 + {theta_4:.2f}x^4', color='red')

plt.legend()
plt.show()

