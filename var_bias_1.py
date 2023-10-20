import matplotlib.pyplot as plt
import scipy.io
import numpy as np


data = scipy.io.loadmat('data.mat')
x = data['x'].flatten()
y = data['y'].flatten()

# plot data
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

# Number of random samples
num_samples = 30
num_iterations = 200

# Initialize an array to store the fitted theta_0 and theta_1 values
theta_0_values = []
theta_1_values = []


for _ in range(num_iterations):
    # Randomly select 30 data samples
    random_indices = np.random.choice(len(x), size=num_samples, replace=False)
    x_sample = x[random_indices]
    y_sample = y[random_indices]
    
    # Perform linear regression (first-order polynomial)
    X = np.column_stack((np.ones_like(x_sample), x_sample))

    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_sample)

    # Extract the coefficients from theta
    theta_0, theta_1, = theta
    

    # Store the parameters
    theta_0_values.append(theta_0)
    theta_1_values.append(theta_1)

def fitted_function(x,i):
    return theta_0_values[i] + theta_1_values[i]*x

mse = []
# Plot the 200 lines
for i in range(num_iterations):
    predict_y = fitted_function(x,i)
    plt.plot(x, predict_y, color='grey', alpha=0.1)
    mse = np.append(mse,np.mean((predict_y - y)**2))

print(np.mean(mse))

plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines')
plt.legend()
plt.show()

