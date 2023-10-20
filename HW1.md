# 資科三 110703065 詹松霖

## 1. 
1.1 Plot the data using plot function.
```python
import scipy.io
import matplotlib.pyplot as plt

# load data
mat_data = scipy.io.loadmat('data.mat')
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()

# plot data
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

plt.show()
```

1.2   
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/least_square_line.py)  
theta_0 = 0.2070272  
theta_1 = 5.98091717

1.3  
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/least_square_poly1.py)  
y = 1.18 + 0.14x + 5.84x^2




