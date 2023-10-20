import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# load data
mat_data = scipy.io.loadmat('data.mat')
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()
n = len(x)
lr = 0.01

theta_0 = 0
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0

for i in range(100000):
    y_pred = theta_0 + theta_1 * x + theta_2 * x**2 + theta_3 * x**3 + theta_4 * x**4
    pd_0 = (1/n) * np.sum(y_pred - y)
    pd_1 = (1/n) * np.sum((y_pred - y) * x)
    pd_2 = (1/n) * np.sum((y_pred - y) * x**2)
    pd_3 = (1/n) * np.sum((y_pred - y) * x**3)
    pd_4 = (1/n) * np.sum((y_pred - y) * x**4)
    theta_0 = theta_0 - lr * pd_0
    theta_1 = theta_1 - lr * pd_1
    theta_2 = theta_2 - lr * pd_2
    theta_3 = theta_3 - lr * pd_3
    theta_4 = theta_4 - lr * pd_4
    

plt.plot(x, y)

y_ls = theta_0 + x*theta_1 + x**2*theta_2 + x**3*theta_3 + x**4*theta_4
plt.plot(x, y_ls, label=f'least square quartic curve: y = {theta_0:.2f} + {theta_1:.2f}x + {theta_2:.2f}x^2 + {theta_3:.2f}x^3 + {theta_4:.2f}x^4', color='red')


def line(v, theta_0, theta_1, theta_2, theta_3, theta_4):
    return theta_0 + theta_1 * v + theta_2 * v**2 + theta_3 * v**3 + theta_4 * v**4
y_pred = line(x, theta_0, theta_1, theta_2, theta_3, theta_4)

# Calculate the mean squared error
mse = np.mean((y - y_pred)**2)

# Print the mean squared error
print(f'Mean Squared Error: {mse}') # 0.010413051044174369

# plot data

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()