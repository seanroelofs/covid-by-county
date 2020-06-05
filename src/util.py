import torch
import time

def evaluate(model, evaluation, loader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            Y_hat = model(X)
            loss += evaluation(Y_hat, Y) * Y_hat.shape[0] 
    loss /= len(loader.dataset)
    model.train()
    return loss.item() 


def train(model, criterion, evaluation, optimizer, train_load, val_load, num_epochs = 1, silent = 0):
    train_criterion = []
    val_criterion = []
    train_evaluation = []
    val_evaluation = []

    epoch_times = []

    if silent != 2:
        print(f'\n Model: {model.__class__.__name__}')
        print(f' Criterion: {criterion.__class__.__name__}')
        print(f' Optimizer: {optimizer.__class__.__name__}')
        print('----------Begin Training----------')

    train_criterion.append(evaluate(model, criterion, train_load))
    val_criterion.append(evaluate(model, criterion, val_load))
    train_evaluation.append(evaluate(model, evaluation, train_load))
    val_evaluation.append(evaluate(model, evaluation, val_load))


    if silent != 2:
        print(f"  Train loss: {train_evaluation[-1]: .1f}")
        print(f"  Val loss: {val_evaluation[-1]: .1f}")


    for epoch in range(num_epochs):
        tic = time.time()
        for i, (X, Y) in enumerate(train_load):
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()

        train_criterion.append(evaluate(model, criterion, train_load))
        val_criterion.append(evaluate(model, criterion, val_load))
        train_evaluation.append(evaluate(model, evaluation, train_load))
        val_evaluation.append(evaluate(model, evaluation, val_load))

        toc = time.time()
        epoch_times.append(toc -tic)
        if silent == 0 or (silent == 1 and  epoch == num_epochs -1):
            print(f"Epoch {epoch+1} executed in {toc - tic: .1f} seconds")
            print(f"  Train loss: {train_evaluation[-1]: .1f}")
            print(f"  Val loss: {val_evaluation[-1]: .1f}")
    if silent != 2:
        print('----------End Training----------\n')
    model.eval()
    return (train_criterion, val_criterion), (train_evaluation, val_evaluation)

