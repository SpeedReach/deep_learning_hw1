from __future__ import print_function
import keras
from keras.datasets import mnist
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# 28*28 => 1*784
x_selected = x_selected.reshape(5000, -1)
# Normalize the data
mean = np.mean(x_selected, axis=0)
std_dev = np.std(x_selected, axis=0)

normalized_images = (x_selected - mean) / (std_dev + 1e-10 )

# Step 3: Compute Covariance Matrix
cov_matrix = np.cov(normalized_images, rowvar=False)

# Step 4: Compute Eigenpairs
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort Eigenpairs
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]



def doPCA(dimension: int):
  # Step 1: Select the top {500} eigenvectors
  top_eigenvectors = sorted_eigenvectors[:, :dimension]

  # Step 2: Project the normalized data onto the top 500 eigenvectors
  reduced_data = np.dot(normalized_images, top_eigenvectors)
  # Step 1: Project back to the original space
  decoded_data = np.dot(reduced_data, top_eigenvectors.T)

  # Step 2: Add back the mean
  decoded_data = (decoded_data * std_dev) + mean
  decoded_data = decoded_data.reshape(5000, 28, 28).astype('float64')

  lines = 10
  columns = 10

  fig = plt.figure()
  fig.suptitle(f'PCA with {dimension} dimensions')
  for i in range(lines):
    for j in range(columns):
      ax = fig.add_subplot(lines,columns, 1+i*10+j)
      plt.imshow(decoded_data[i*500+j,:,:], cmap='binary')
      plt.sca(ax)
      ax.set_xticks([], [])
      ax.set_yticks([], [])

  
doPCA(500)
doPCA(300)
doPCA(100)
doPCA(50)
plt.show()