# 資科三 110703065 詹松霖
All models are implmented by myself.
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
## Curves
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/var_bias_curves.png?raw=true)
### Bias:
The quartic model is more complex and can capture more intricate patterns in the data. It is able to fit the data more closely, indicating **lower** bias.


## Lines
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/var_bias_lines.png?raw=true)
### Bias:
The linear model has a simpler form, which may not be able to capture the complexity of the underlying data if it follows a more curved pattern. This leads to **higher** bias, as the model is less flexible and might not fit the data well.

## Variance
I'm not quite sure how to determine which model has a higher variance, because quartic curves are densely concentrated in the middle portion, so if we only focus on the middle, the variance appears to be smaller than the lines. However, if we take a step back and examine the edges, quartic curves are more spread out. If we only consider the tails, then the variance is greater for quartic curves compared to the lines.
After doing some research online, I've come to understand that this is a concept known as "local smoothness" versus "global smoothness". Different models may perform differently in different parts of the data. When it comes to choosing models, it might be necessary to consider the purpose of the model and focus on the segments of interest, such as the middle portion.

# 3
[code](https://github.com/SpeedReach/deep_learning_hw1/blob/main/logistic_reg2.py)  
Test Accuracy: 0.97  

**decision boundary**:  
theta0: -0.060600000000000084  
theta1: 0.10332857142857141  
![image](https://github.com/SpeedReach/deep_learning_hw1/blob/main/images/logistic_refression.png?raw=true)