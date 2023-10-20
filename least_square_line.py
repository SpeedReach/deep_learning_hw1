import numpy as np
import scipy.io

# Load the data from data.mat
data = scipy.io.loadmat('data.mat')

# Extract x and y from the loaded data
x = data['x']
y = data['y']

X = np.column_stack((np.ones_like(x), x,x**2))

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


print(theta)

# Extract the coefficients from theta
theta_0, theta_1, = theta

# Define the fitted parabola function
fitted_parabola = lambda x: theta_0 + theta_1*x

# Evaluate the fitted parabola on the x values
y_fitted = fitted_parabola(x)

import matplotlib.pyplot as plt

plt.scatter(x, y, label='Data')
plt.plot(x, y_fitted, label='Fitted Line', color='red')
plt.legend()
plt.show()
