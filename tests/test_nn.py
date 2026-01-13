import random

from tinygrad_scratch.nn import MLP, mse_loss
from tinygrad_scratch.optim import Adam


def test_overfit_tiny_batch():
    random.seed(0)
    xs = [[2.0, 3.0], [3.0, -1.0], [-1.0, 0.5], [-2.0, -2.0]]
    ys = [1.0, 1.0, -1.0, -1.0]
    model = MLP(2, [8, 1])
    opt = Adam(model.parameters(), lr=0.05)
    loss = None
    for _ in range(200):
        preds = [model(x) for x in xs]
        loss = mse_loss(preds, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.data < 0.01
