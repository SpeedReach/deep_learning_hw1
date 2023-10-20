import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random

# Load train data
train_data = scipy.io.loadmat('train.mat')
X_train = np.hstack((train_data['x1'],train_data['x2']))  # 2-D points
Y_train = train_data['y'].ravel()  # corresponding labels

lr = 0.01

w0 = random.randint(-100,100)/100.0
w1 = random.randint(-100,100)/100.0

identify = lambda t: t > 0.5 and np.array(1) or np.array(0)

def cal_prob(x1, x2):
    return 1/(1 + np.exp(-(w0*x1 + w1*x2)))

# Gradient Descent
for i in range(100000):
    y_pred = cal_prob(X_train.T[0], X_train.T[1])
    y_pred = np.array([ identify(yi) for yi in y_pred])
    pd_0 = (1/len(X_train)) * np.sum((y_pred - Y_train) * X_train.T[0])
    pd_1 = (1/len(X_train)) * np.sum((y_pred - Y_train) * X_train.T[1])
    w0 = w0 - lr * pd_0
    w1 = w1 - lr * pd_1
    
# Load test data
test_data = scipy.io.loadmat('test.mat')
X_test = np.hstack((test_data['x1'],test_data['x2']))  # 2-D points
Y_test = test_data['y'].ravel()  # corresponding labels

# Predict test data
pred = cal_prob(X_test.T[0], X_test.T[1])
pred = np.array([ identify(yi) for yi in pred])


# Calculate the test error rate
test_len = len(Y_test)
matched = 0
for i in range(test_len):
    if pred[i] == Y_test[i]:
        matched += 1

print(f'Test Accuracy: {matched/test_len:.2f}')
print(f'w0: {w0}')
print(f'w1: {w1}')

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
        
        
plt.plot(X_train[:,0], -(w0*X_train[:,0])/w1, color='black')
plt.show()
