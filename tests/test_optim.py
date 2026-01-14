from tinygrad_scratch.engine import Value
from tinygrad_scratch.optim import SGD, Adam, Momentum


def minimize(optimizer_class, **kwargs):
    x = Value(0.0)
    opt = optimizer_class([x], **kwargs)
    for _ in range(500):
        loss = (x - 3.0) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.data


def test_sgd_minimizes():
    assert abs(minimize(SGD, lr=0.1) - 3.0) < 1e-2


def test_momentum_minimizes():
    assert abs(minimize(Momentum, lr=0.05) - 3.0) < 1e-2


def test_adam_minimizes():
    assert abs(minimize(Adam, lr=0.1) - 3.0) < 1e-2
