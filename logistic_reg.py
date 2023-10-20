import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load train data
train_data = scipy.io.loadmat('train.mat')
X_train = np.hstack((train_data['x1'],train_data['x2']))  # 2-D points
Y_train = train_data['y']  # corresponding labels

# Load test data
test_data = scipy.io.loadmat('test.mat')
X_test = np.hstack((test_data['x1'],test_data['x2']))  # 2-D points
Y_test = test_data['y']  # corresponding labels

from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
log_reg = LogisticRegression().fit(X_train, Y_train.ravel())


score = log_reg.score(X_test, Y_test.ravel())
print(f'Test Accuracy: {score:.2f}')

b = log_reg.intercept_[0]
w1, w2 = log_reg.coef_.T

print(f'w1: {w1}')
print(f'w2: {w2}')

# Disicion boundary = b + w1*x1 + w2*x2 = 0
plt.plot(X_test[:,0], -(b + w1*X_test[:,0])/w2, color='black')

for i in range(len(Y_test)):
    if Y_test[i] == 1:
        plt.scatter(X_test[i,0], X_test[i,1], color='red')
    else:
        plt.scatter(X_test[i,0], X_test[i,1], color='blue')
for i in range(len(Y_train)):
    if Y_train[i] == 1:
        plt.scatter(X_train[i,0], X_train[i,1], color='yellow')
    else:
        plt.scatter(X_train[i,0], X_train[i,1], color='green')

plt.show()

