import random

import numpy as np
from sklearn.datasets import make_moons

from tinygrad_scratch.nn import MLP, mse_loss
from tinygrad_scratch.optim import Adam


def load_moons(n=200, noise=0.2, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    targets = [1.0 if label == 1 else -1.0 for label in y]
    return X.tolist(), targets


def accuracy(model, xs, targets):
    correct = 0
    for x, t in zip(xs, targets):
        pred = model(x).data
        if (pred > 0) == (t > 0):
            correct += 1
    return correct / len(xs)


def train(epochs=120, lr=0.05, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    xs, targets = load_moons(seed=seed)
    model = MLP(2, [16, 16, 1])
    opt = Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        preds = [model(x) for x in xs]
        loss = mse_loss(preds, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.data)
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {accuracy(model, xs, targets):.3f}")
    print(f"final accuracy {accuracy(model, xs, targets):.3f}")
    return model, xs, targets, history


if __name__ == "__main__":
    train()
