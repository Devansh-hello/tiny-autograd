import pytest

from tinygrad_scratch.engine import Value

torch = pytest.importorskip("torch")


def max_grad_diff(expr_ours, expr_torch, inputs):
    ours_vals = [Value(x) for x in inputs]
    out = expr_ours(ours_vals)
    out.backward()
    ours = [v.grad for v in ours_vals]

    torch_vals = [torch.tensor(x, dtype=torch.double, requires_grad=True) for x in inputs]
    tout = expr_torch(torch_vals)
    tout.backward()
    theirs = [t.grad.item() for t in torch_vals]

    return max(abs(a - b) for a, b in zip(ours, theirs))


def test_parity_arithmetic():
    ours = lambda v: v[0] * v[1] + v[0] / v[1] - v[1] ** 3
    ref = lambda t: t[0] * t[1] + t[0] / t[1] - t[1] ** 3
    assert max_grad_diff(ours, ref, [2.0, -3.0]) < 1e-6


def test_parity_activations():
    ours = lambda v: (v[0].tanh() + v[1].relu()) * v[2].exp()
    ref = lambda t: (t[0].tanh() + t[1].relu()) * t[2].exp()
    assert max_grad_diff(ours, ref, [0.5, 1.2, -0.3]) < 1e-6


def test_parity_log_and_pow():
    ours = lambda v: (v[0] ** 2 + 1.0).log() + v[1] ** 4
    ref = lambda t: (t[0] ** 2 + 1.0).log() + t[1] ** 4
    assert max_grad_diff(ours, ref, [1.5, 0.8]) < 1e-6
