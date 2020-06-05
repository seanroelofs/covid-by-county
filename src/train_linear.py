import sys
sys.path.append('../')
from models.linear_model import Linear
from util import train, evaluate
from data.loaders import dataset_loaders
import torch
import matplotlib.pyplot as plt

train_load, val_load, test_load = dataset_loaders()

model = Linear()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.05)
criterion = torch.nn.MSELoss()
evaluation = torch.nn.L1Loss()
losses, evals = train(model, criterion, evaluation, optimizer, train_load, val_load, 20, silent = 2)

plt.figure()
plt.title("Linear Regression")
plt.plot(losses[0], label = "Train Loss")
plt.plot(losses[1], label = "Val Loss")
plt.legend()
plt.savefig("../output/linear/train_val_loss")

print("Loss:")
print("Train: ", losses[0][-1])
print("Val: ", losses[1][-1])
print("Test: ", evaluate(model, criterion, test_load))

print("Evaluation")
print("Train: ", evals[0][-1])
print("Val: ", evals[1][-1])
print("Test: ", evaluate(model, evaluation, test_load))



