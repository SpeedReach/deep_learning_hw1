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
theta_2_values = []
theta_3_values = []
theta_4_values = []

for _ in range(num_iterations):
    # Randomly select 30 data samples
    random_indices = np.random.choice(len(x), size=num_samples, replace=False)
    x_sample = x[random_indices]
    y_sample = y[random_indices]
    
    X = np.column_stack((np.ones_like(x_sample), x_sample,x_sample**2, x_sample**3,x_sample**4))

    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_sample)
    
    # Extract the parameters
    theta_0, theta_1, theta_2, theta_3,theta_4  = theta
    
    # Store the parameters
    theta_0_values.append(theta_0)
    theta_1_values.append(theta_1)
    theta_2_values.append(theta_2)
    theta_3_values.append(theta_3)
    theta_4_values.append(theta_4)


def fitted_function(x,i):
    return theta_0_values[i] + theta_1_values[i]*x + theta_2_values[i]*x**2 + theta_3_values[i]*x**3 + theta_4_values[i]*x**4

mse = []
# Plot the 200 lines
for i in range(num_iterations):
    predict_y = fitted_function(x,i)
    plt.plot(x, predict_y, color='grey', alpha=0.1)
    mse = np.append(mse,np.mean((predict_y - y)**2))

print(np.mean(mse))

plt.xlabel('x')
plt.ylabel('y')
plt.title('Quartic curves')
plt.legend()
plt.show()