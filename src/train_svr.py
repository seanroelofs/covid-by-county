import sys
sys.path.append('../')
from data.loaders import numpy_data
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

X_train, Y_train, X_val, Y_val, X_test, Y_test = numpy_data()

space = (-3, 0, -2, 1)

extent = (space[0] - 0.5 * (space[1] - space[0])/5, space[1] + 0.5 * (space[1] - space[0])/5, space[2] - 0.5 * (space[3] - space[2])/5, space[3] + 0.5 * (space[3] - space[2])/5)

Cs = np.logspace(space[2], space[3], num = 6, endpoint = True)
epsilons  = np.logspace(space[0], space[1], num =6, endpoint = True)

train_errs = np.empty((len(Cs), len(epsilons)))
val_errs = np.empty((len(Cs), len(epsilons)))

best_svr = None
best_val_err = 10000

for i in range(len(Cs)):
    for j in range(len(epsilons)):
        svr = SVR(C = Cs[i], epsilon = epsilons[j], kernel = 'linear')
        svr.fit(X_train, Y_train.ravel())
        train_err = np.absolute(svr.predict(X_train)[:,np.newaxis] - Y_train).mean()
        val_err = np.absolute(svr.predict(X_val)[:,np.newaxis] - Y_val).mean()
        if val_err < best_val_err:
            best_val_err = val_err
            best_svr = svr
            print(epsilons[j], Cs[i])
        train_errs[i, j] = train_err
        val_errs[i, j] = val_err

plt.figure()
plt.imshow(val_errs[::-1], cmap = "Greys", interpolation = 'nearest', extent = extent)
plt.ylabel('$\log(C)$')
plt.xlabel('$\log(\epsilon)$')
plt.title('Hyperparameter Search \n White is lower error')
plt.savefig("../output/svr/val_hyperparameter.png")


print("Train: ", np.min(train_errs))
print("Val: ", np.min(val_errs))
print("Test: ", np.absolute(best_svr.predict(X_test)[:,np.newaxis] - Y_test).mean())
