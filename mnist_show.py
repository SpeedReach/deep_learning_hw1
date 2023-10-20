from __future__ import print_function
import keras
from keras.datasets import mnist
import random
import numpy as np

# input image dimensions 28x28 img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

train_len = len(x_train)
indexs = list(range(0, train_len))
random.shuffle(indexs)

shuffled_x_train = x_train[indexs]
shuffled_y_train = y_train[indexs]
x_train = shuffled_x_train
y_train = shuffled_y_train

#pick 500 samples for each digit from the training set
selected_indices = []
for i in range(10):
    indices = np.where(y_train == i)[0][:500]
    selected_indices.extend(indices)

x_selected = x_train[selected_indices]



import numpy as np
import matplotlib.pyplot as plt
amount= 50
lines = 5
columns = 10
number = np.zeros(amount)

showIndex = np.random.randint(0, 5000, amount)
x_selected = x_selected[showIndex]


for i in range(amount):
  print(int(showIndex[i]/500))

fig = plt.figure()
  
for i in range(amount):
  ax = fig.add_subplot(lines, columns, 1 + i)
  plt.imshow(x_selected[i,:,:], cmap='binary')
  plt.sca(ax)
  ax.set_xticks([], [])
  ax.set_yticks([], [])
plt.show()