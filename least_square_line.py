import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the data from data.mat
data = scipy.io.loadmat('data.mat')

# Extract x and y from the loaded data
x = data['x'].flatten()
y = data['y'].flatten()

# plot data
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

X = np.column_stack((np.ones_like(x), x,))

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Extract the coefficients from theta
theta_0, theta_1, = theta

# Define the fitted function
fitted_parabola = lambda x: theta_0 + theta_1*x

# Evaluate the fitted parabola on the x values
y_fitted = fitted_parabola(x)
line =  f'y = {theta_0:.2f} + {theta_1:.2f}x'
plt.plot(x, y_fitted, label=line, color='red')
plt.legend()


#Calculate the mean squared error
mse = np.mean((y_fitted - y)**2)
print(f'mse = {mse}')


plt.show()
