import numpy as np
from util import get_data
import matplotlib.pyplot as plt

headers, X, Y = get_data()
X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
m, n = X.shape
shuffle_indices = np.arange(m)
np.random.shuffle(shuffle_indices)

X = X[shuffle_indices]
Y = Y[shuffle_indices]
X_train, Y_train = X[:m*3//5], Y[:m*3//5]
X_val, Y_val = X[m*3//5:m*4//5], Y[m*3//5:m*4//5]
X_test, Y_test = X[m*4//5:], Y[m*4//5:]


W = np.zeros((n, 1))
B = np.zeros((1, 1))
learning_rate = 0.1
reg = 0.01
train_loss = []
val_loss = []
iters = 50
for i in range(iters):
    Y_train_pred = X_train @ W + B
    Y_val_pred = X_val @ W + B
    print(Y_train_pred.shape, Y_train.shape)
    train_loss.append(np.sum(np.square(Y_train_pred - Y_train))/len(X_train) + reg * np.sum(W*W)/2) 
    val_loss.append(np.sum(np.square(Y_val_pred - Y_val))/len(X_val) + reg * np.sum(W*W)/2)
    dW = X_train.T @ (Y_train_pred - Y_train)/len(X_train) + reg * W
    W -= learning_rate * dW
    dB = np.sum(Y_train_pred - Y_train)/len(X_train)
    B -= learning_rate * dB

plt.figure()
plt.plot(np.arange(iters), train_loss, label = "train")
plt.plot(np.arange(iters), val_loss, label = "validation")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig("../output/linear_classifier")
print(val_loss[-1])



