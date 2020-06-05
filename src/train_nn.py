import sys
sys.path.append('../')
from models.neural_network import NN
from util import train, evaluate
from data.loaders import dataset_loaders
import torch
import numpy as np
import matplotlib.pyplot as plt

train_load, val_load, test_load = dataset_loaders()

model = NN([128, 128, 128, 32])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, weight_decay = 0)
criterion = torch.nn.MSELoss()
evaluation = torch.nn.L1Loss()
losses, errs  = train(model, criterion, evaluation, optimizer, train_load, val_load, num_epochs = 25, silent = 2)


test_loss = evaluate(model, criterion, test_load)
test_errs = evaluate(model, evaluation, test_load)

plt.figure()
plt.plot(losses[0], label = "Train Loss")
plt.plot(losses[1], label = "Val Loss")
plt.legend()
#plt.savefig("../output/nn/train_val_loss")

print("Train: ", losses[0][-1])
print("Val: ", losses[1][-1])
print("Test: ", np.array(test_loss).mean())
print("Train: ", errs[0][-1])
print("Val: ", errs[1][-1])
print("Test: ", np.array(test_errs).mean())



