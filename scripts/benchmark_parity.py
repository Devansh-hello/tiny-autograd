import random

import torch

from tinygrad_scratch.engine import Value


def expr(a, b, c):
    return (a * b + c).tanh() * (a - c).relu() + (b * c).exp() / (a * a + 1.0)


def run(trials=500):
    random.seed(0)
    worst = 0.0
    for _ in range(trials):
        inputs = [random.uniform(-2, 2) for _ in range(3)]

        vs = [Value(x) for x in inputs]
        out = expr(vs[0], vs[1], vs[2])
        out.backward()
        ours = [v.grad for v in vs]

        ts = [torch.tensor(x, dtype=torch.double, requires_grad=True) for x in inputs]
        tout = expr(ts[0], ts[1], ts[2])
        tout.backward()
        theirs = [t.grad.item() for t in ts]

        diff = max(abs(o - t) for o, t in zip(ours, theirs))
        worst = max(worst, diff)

    print(f"ran {trials} random expressions")
    print(f"worst max|grad difference| vs pytorch: {worst:.2e}")
    return worst


if __name__ == "__main__":
    run()
