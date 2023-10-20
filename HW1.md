# 資科三 110703065 詹松霖

## 1. 
### 1.1 Plot the data using plot function.
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
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/plot_data.jpg?raw=true)

### 1.2   
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/least_square_line.py)  
theta_0 = 0.2070272  
theta_1 = 5.98091717  
mse = 0.20580596682517396
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/lsql.jpg?raw=true)

### 1.3  
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/least_square_poly1.py)  
y = 1.18 + 0.14x + 5.84x^2  
mse = 0.015744919931207565
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/lsqpl1.jpg?raw=true)

### 1.4
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/least_square_poly2.py)  
y = 1.03 + 1.59x + 3.86x^2 + -1.97x^3 + 2.88x^4  
mse = 0.010413051044174356
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/lsqpl2.png?raw=true)

### 1.5
The model with the smallest mean square error fits the dataset the best, which makes it more suitable than the others.
The quartic curve has the smallest mean square error with 0.01, so it's the most suitable.

# 2
## 2.1