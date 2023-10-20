import scipy.io
import matplotlib.pyplot as plt

mat_data = scipy.io.loadmat('data.mat')
x = mat_data['x']
y = mat_data['y']

# plot data
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')


plt.legend()
plt.show()