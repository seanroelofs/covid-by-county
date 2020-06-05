import numpy as np
import csv

def get_numpy_data():
    with open("combined.csv") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = np.array([row for row in reader])
    countys = data[:, 2]
    data = np.delete(data, 2, axis = 1)
    data = data.astype('float64')
    return headers, countys, data[:, :-1], data[:, -1][:,np.newaxis]


headers, countys, X, Y = get_numpy_data()

m, n = X.shape
print(m)
shuffle_indices = np.arange(m)


np.random.seed(0)
np.random.shuffle(shuffle_indices)
X = X[shuffle_indices]
Y = Y[shuffle_indices]

X_train = X[:m*3//5]
std = np.std(X_train, axis = 0)
mean = np.mean(X_train, axis = 0)
X = (X - mean) / std

X_train, Y_train = X[:m*3//5], Y[:m*3//5]
X_val, Y_val = X[m*3//5:m*4//5], Y[m*3//5:m*4//5]
X_test, Y_test = X[m*4//5:], Y[m*4//5:]

np.savetxt("train/x", X_train)
np.savetxt("train/y", Y_train)

np.savetxt("val/x", X_val)
np.savetxt("val/y", Y_val)

np.savetxt("test/x", X_test)
np.savetxt("test/y", Y_test)




