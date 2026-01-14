import random

import numpy as np

from scripts.train_moons import accuracy, load_moons
from tinygrad_scratch.nn import MLP, mse_loss
from tinygrad_scratch.optim import Adam


def test_moons_accuracy_above_95():
    random.seed(0)
    np.random.seed(0)
    xs, targets = load_moons(n=40, seed=0)
    model = MLP(2, [8, 8, 1])
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        preds = [model(x) for x in xs]
        loss = mse_loss(preds, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert accuracy(model, xs, targets) > 0.95
